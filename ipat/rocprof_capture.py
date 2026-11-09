"""Build the bounded launch wrapper from the pinned profiling-archive recipe."""
REFERENCE = ('https://github.com/ajith-sirra-amd/InferenceMAX_Rocm_Team/blob/'
             'f84d42eb3e6361888eff8dde16e2cc0a395f252a/upstream/InferenceX/benchmarks/'
             'single_node/agentic/patches/kimik3_fp4_mi355x_mtp.profiling-archive.sh#L580-L586')


def launch_command(server, settings, output, help_text):
    delay = settings.get('launch_delay_seconds')
    seconds = settings.get('seconds')
    if type(delay) is not int or delay < 0 or type(seconds) is not int or not 1 <= seconds <= 180:
        raise ValueError('rocprof launch needs a calibrated nonnegative launch_delay_seconds and 1–180 capture seconds')
    domains = ['--kernel-trace', '--stats']
    for field, flag in (('hip_trace', '--hip-trace'), ('marker_trace', '--marker-trace')):
        if settings.get(field):
            domains.append(flag)
    for flag in [*domains, '--collection-period', '--collection-period-unit', '--summary-output-file']:
        if flag not in help_text:
            raise RuntimeError('Installed rocprofv3 does not advertise required option ' + flag)
    return ['rocprofv3', *domains, '-d', str(output / 'traces'), '-f', 'csv',
            '--summary-output-file', str(output / 'rocprof-summary.txt'),
            '--collection-period', f'{delay}:{seconds}:1', '--collection-period-unit', 'sec', '--', *server]
