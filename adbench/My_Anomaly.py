
# PolyU_newADBench project
from adbench.run import RunPipeline
pipeline = RunPipeline(suffix='ADBench', parallel='supervise', realistic_synthetic_mode=None, noise_type='irrelevant_features')
results = pipeline.run()
