"""Install observational hooks after native worker device initialization."""


class DiagnosticWorkerExtension:
    def init_worker_extension(self):
        # Native WorkerWrapperBase invokes this after GPUModelRunner construction
        # and before model loading or graph capture. Keep module import lightweight:
        # Python startup and its helper subprocesses must not install these hooks.
        import diagnostic_hooks

        diagnostic_hooks.install()
        if not diagnostic_hooks.INSTALLED:
            raise RuntimeError("Diagnostic hooks did not finish installing")
