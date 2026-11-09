import copy
import csv
import gzip
import json
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from unittest.mock import patch, MagicMock

from kernel_report import aggregate, sha
from rocprof_capture import launch_command
from container import profile, stop
from runner import validate_recipe, resolve_pass_recipe
from test_runner import RECIPE


class KernelReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.traces = self.root / 'traces'; self.traces.mkdir()
        self.output = self.root / 'report'; self.output.mkdir()

    def trace(self, events, name='rank.trace.json.gz'):
        path = self.traces / name
        with gzip.open(path, 'wt') as handle:
            json.dump({'traceEvents': events, 'displayTimeUnit': 'ms'}, handle)
        return path

    def kernel(self, name='gemm', duration=10, device=0, external=7, **extra):
        return {'ph':'X', 'cat':'kernel', 'name':name, 'ts':100, 'dur':duration,
                'pid':device, 'tid':1, 'args':{'device':device, 'External id':external, **extra}}

    def test_all_symbols_sorted_population_statistics_and_per_call_exports(self):
        events = [self.kernel(duration=x) for x in (10, 20, 30)]
        events += [self.kernel('small'+str(i), 0.1) for i in range(61)]
        self.trace(events)
        report = aggregate(self.traces, self.output, expected_devices=1)
        row = report['kernels'][0]
        self.assertEqual(report['unique_kernels'], 62)
        self.assertEqual(row['calls'], 3)
        self.assertEqual((row['avg_us'],row['min_us'],row['max_us']), (20,10,30))
        self.assertAlmostEqual(row['std_us'], math.sqrt(200/3))
        self.assertAlmostEqual(sum(k['time_pct'] for k in report['kernels']), 100)
        self.assertEqual(len(report['top10']),10)
        self.assertIsNone(row['iteration_totals'])
        with gzip.open(self.output/'kernel-dispatches.csv.gz','rt') as handle:
            self.assertEqual(len(list(csv.DictReader(handle))),64)
        self.assertEqual(report['top10'][0]['variants'][0]['analysis']['path']['status'],'insufficient_evidence')
        self.assertFalse(any(self.output.glob('benchmark-kernels-*')))

    def test_precise_csv_timestamps_and_overlap_do_not_become_walltime(self):
        path = self.traces/'rank_kernel_trace.csv'
        origin=1790000000000000000
        with path.open('w',newline='') as handle:
            writer=csv.writer(handle)
            writer.writerow(['Kernel_Name','Start_Timestamp','End_Timestamp','Agent_Id','Process_Id'])
            writer.writerow(['quoted, kernel',origin,origin+1234,0,7])
            writer.writerow(['quoted, kernel',origin,origin+2000,1,8])
        r=aggregate(self.traces,self.output,expected_devices=2)
        self.assertAlmostEqual(r['kernels'][0]['total_us'],3.234)
        self.assertEqual(len(r['per_device']),2)
        self.assertTrue(r['device_coverage_complete'])
        self.assertIn('not wall time',r['note'])

    def test_operator_correlation_and_iteration_are_trace_local_and_shape_specific(self):
        step={'ph':'X','name':'ProfilerStep#1','ts':0,'dur':30,'pid':99,'tid':4}
        op={'ph':'X','cat':'cpu_op','name':'aten::mm','ts':1,'dur':2,'pid':99,'tid':4,
            'args':{'External id':7,'Input Dims':[[2,4],[4,8]],'Input type':['BF16','BF16']}}
        self.trace([step,op,self.kernel(duration=10),self.kernel(duration=20)])
        other=copy.deepcopy(op);other['args']['Input Dims']=[[4,4],[4,8]]
        self.trace([step,other,self.kernel(duration=40)],'rank2.trace.json.gz')
        r=aggregate(self.traces,self.output)
        self.assertEqual(len(r['top10'][0]['variants']),2)
        self.assertEqual(r['coverage']['dispatches_with_operator_link'],3)
        self.assertEqual(r['coverage']['dispatches_with_iteration'],3)
        stats=r['kernels'][0]['iteration_totals']
        self.assertEqual(stats['calls'],2)
        self.assertEqual(stats['avg_us'],35)
        self.assertEqual(stats['std_us'],5)
        self.assertEqual(r['top10'][0]['variants'][0]['analysis']['tensors']['status'],'unavailable')

    def test_ambiguous_external_ids_are_not_arbitrarily_assigned(self):
        op={'ph':'X','cat':'cpu_op','name':'op','ts':1,'dur':2,'pid':99,'args':{'External id':7,'Input Dims':[[2]]}}
        self.trace([op,op,self.kernel()])
        r=aggregate(self.traces,self.output)
        self.assertEqual(r['coverage']['dispatches_with_operator_link'],0)
        self.assertNotIn('operator_input_shapes',r['top10'][0]['variants'][0])

    def test_malformed_trace_keeps_partial_evidence_and_marks_errors(self):
        path=self.trace([self.kernel(),self.kernel(duration=-1)])
        r=aggregate(self.traces,self.output)
        self.assertEqual(r['kernel_count'],1)
        self.assertTrue(r['errors'])
        path.unlink()
        (self.traces/'bad.trace.json').write_text('{"traceEvents":[')
        r=aggregate(self.traces,self.output)
        self.assertEqual(r['kernel_count'],0)
        self.assertTrue(r['errors'])

    def test_duplicate_exports_are_not_added_and_format_selection_is_explicit(self):
        path=self.trace([self.kernel()])
        (self.traces/'copy.trace.json.gz').write_bytes(path.read_bytes())
        r=aggregate(self.traces,self.output)
        self.assertEqual(r['kernel_count'],1)
        self.assertTrue(r['warnings'])
        (self.traces/'rank_kernel_trace.csv').write_text('Kernel_Name,Start_Timestamp,End_Timestamp,Agent_Id\ngemm,0,2000,0\n')
        r=aggregate(self.traces,self.output)
        self.assertEqual(r['format'],'rocprof')
        self.assertEqual(r['kernels'][0]['total_us'],2)
        self.assertTrue(r['warnings'])

    def make_evidence(self):
        self.trace([self.kernel()])
        report=aggregate(self.traces,self.output)
        v=report['top10'][0]['variants'][0]
        (self.output/'counter.txt').write_text('Synthetic unit-test evidence; not GPU measurements.')
        ref={'path':'counter.txt','sha256':sha(self.output/'counter.txt')}
        binding={'image_digest':'sha256:synthetic','code_object_sha256':'synthetic',
            'runtime_backend':'test_backend','shape_signature':v['shape_signature'],'device':'0','phase':'decode','graph_mode':'replay','source':ref}
        evidence={'capture_sha256':report['capture_sha256'],'variant_id':v['variant_id'],'binding':binding,
            'roofline':{'source':ref,'calibration_source':ref,'scope':'single_device_matched_dispatch_population',
                'measurement_id':'counter-pass-dispatch-1','flop_definition':'Executed FLOPs; FMA=2',
                'hbm_byte_definition':'Measured read + write bytes','compute_precision':'BF16',
                'duration_us':10,'calls':1,'flops':1000000,'hbm_bytes':10000,'compute_ceiling_tflops':1,
                'hbm_ceiling_gbs':10,'ceiling_kind':'measured'}}
        return v,evidence,ref

    def test_matched_roofline_formulas_and_export_workbook(self):
        variant,evidence,_=self.make_evidence()
        (self.output/'kernel-evidence.json').write_text(json.dumps({variant['variant_id']:evidence}))
        r=aggregate(self.traces,self.output)
        roof=r['top10'][0]['variants'][0]['analysis']['roofline']
        self.assertEqual(roof['status'],'computed_from_supplied_evidence')
        self.assertEqual(roof['arithmetic_intensity_flops_per_byte'],100)
        self.assertAlmostEqual(roof['achieved_tflops'],.1)
        self.assertAlmostEqual(roof['achieved_hbm_gbs'],1)
        self.assertAlmostEqual(roof['roof_utilization_pct'],10)
        self.assertIn('<svg', (self.output/'kernel-report.html').read_text())
        with zipfile.ZipFile(self.output/'kernel-report.xlsx') as archive:
            for file in archive.namelist(): ET.fromstring(archive.read(file))
            xml=archive.read('xl/worksheets/sheet1.xml').decode()
            for heading in ('Precision Evidence','Std (us)','GPU Duration Share %'):self.assertIn(heading,xml)

    def test_mismatched_or_tampered_counter_evidence_never_gets_roofline(self):
        variant,evidence,_=self.make_evidence()
        evidence['capture_sha256']='another-run'
        (self.output/'kernel-evidence.json').write_text(json.dumps({variant['variant_id']:evidence}))
        r=aggregate(self.traces,self.output)
        a=r['top10'][0]['variants'][0]['analysis']
        self.assertEqual(a['roofline']['status'],'unavailable')
        self.assertTrue(a['errors'])
        evidence['capture_sha256']=r['capture_sha256'];evidence['roofline']['scope']='all-gpus-summed'
        (self.output/'kernel-evidence.json').write_text(json.dumps({variant['variant_id']:evidence}))
        self.assertEqual(aggregate(self.traces,self.output)['top10'][0]['variants'][0]['analysis']['roofline']['status'],'unavailable')

    def test_optimized_verdict_requires_correct_compatible_matched_alternative(self):
        variant,evidence,ref=self.make_evidence()
        row={'implementation':'current','source':ref,'binding':evidence['binding'],'correctness_passed':True,
             'compatible':True,'repeats':3,'avg_us':20,'min_us':19,'max_us':21}
        candidate=dict(row,implementation='candidate',avg_us=10,min_us=9,max_us=11)
        evidence['path']={'source':ref,'dispatch_source':ref,'selected_implementation':'current','tested_candidates':[row,candidate]}
        def analyze():
            (self.output/'kernel-evidence.json').write_text(json.dumps({variant['variant_id']:evidence}))
            return aggregate(self.traces,self.output)['top10'][0]['variants'][0]['analysis']['path']['status']
        self.assertEqual(analyze(),'faster_tested_path_available')
        candidate['correctness_passed']=False
        self.assertEqual(analyze(),'insufficient_evidence')
        candidate.update(correctness_passed=True,min_us=18,avg_us=20,max_us=22)
        self.assertEqual(analyze(),'inconclusive_timing_overlap')

    def test_html_and_excel_do_not_execute_kernel_names(self):
        self.trace([self.kernel('<script>alert(1)</script>|=1+1')])
        aggregate(self.traces,self.output)
        page=(self.output/'kernel-report.html').read_text()
        self.assertNotIn('<script>',page)
        self.assertIn('&lt;script&gt;',page)
        with zipfile.ZipFile(self.output/'kernel-report.xlsx') as archive:
            self.assertNotIn('<f>',archive.read('xl/worksheets/sheet1.xml').decode())


class RocprofLaunchTests(unittest.TestCase):
    def settings(self):
        return {'kind':'rocprof','mode':'launch','seconds':60,'delay_seconds':30,'launch_delay_seconds':500,
                'start_marker':'Phase profiling (profiling) started'}

    def test_command_preserves_server_and_uses_calibrated_window(self):
        settings=self.settings();server=['vllm','serve','org/Model','--config','value with spaces']
        help_text='--kernel-trace --stats --collection-period --collection-period-unit --summary-output-file --hip-trace --marker-trace'
        cmd=launch_command(server,settings,Path('/output/profile'),help_text)
        self.assertEqual(cmd[cmd.index('--')+1:],server)
        self.assertIn('500:60:1',cmd)
        self.assertNotIn('--hip-trace',cmd)
        settings.update(hip_trace=True,marker_trace=True)
        self.assertIn('--hip-trace',launch_command(server,settings,Path('/output'),help_text))
        with self.assertRaises(RuntimeError):launch_command(server,settings,Path('/output'),'--kernel-trace')

    def test_calibration_preserves_frozen_recipe_and_requires_completed_baseline(self):
        settings=self.settings();settings.pop('launch_delay_seconds')
        recipe=dict(RECIPE,profile=settings)
        records=[{'pass':'baseline','status':'completed','measurement_start_after_server_seconds':2176.25}]
        resolved=resolve_pass_recipe(recipe,'profile',records)
        self.assertEqual(resolved['profile']['launch_delay_seconds'],2207)
        self.assertNotIn('launch_delay_seconds',recipe['profile'])
        self.assertEqual(resolve_pass_recipe(recipe,'baseline',[]),recipe)
        records[0]['status']='failed'
        with self.assertRaises(ValueError):resolve_pass_recipe(recipe,'profile',records)

    def test_launch_validation_and_warmup_overlap_rejection(self):
        settings=self.settings()
        validate_recipe(dict(RECIPE,profile=settings))
        with self.assertRaises(ValueError):validate_recipe(dict(RECIPE,profile=dict(settings,start_marker='')))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);log=root/'client/logs/aiperf.log';log.parent.mkdir(parents=True);log.write_text(settings['start_marker'])
            result={};finished=MagicMock();finished.wait.return_value=False
            with patch('container.time.monotonic',return_value=800):
                profile(dict(RECIPE,profile=settings),MagicMock(),root,'http://localhost',result,finished,server_started=100)
            self.assertEqual(result['status'],'failed')
            self.assertIn('recalibrate',result['error'])
            finished.wait.assert_not_called()
