results = []
with open("xla-mobilenet-second-time-wo-cache.log", "r") as in_file:
  with open("filter-mobilenet-second-time-wo-cache.log", "w") as out_file:
    for line in in_file:
      if line.find("OptimizedHloModuleCache::MaybeLoadOptimizedModule for module") != -1:
        out_file.write(line)
      if line.find("GpuCompiler::RunHloPasses for module") != -1:
        out_file.write(line)
      if line.find("GpuCompiler::OptimizeHloModule optimization pass pipeline for module") != -1:
        out_file.write(line)
      if line.find("GpuCompiler::OptimizeHloModule layout assignment pass pipeline for module") != -1:
        out_file.write(line)
      if line.find("GpuCompiler::OptimizeHloModule post layout assignment pass pipeline for module") != -1:
        out_file.write(line)
      if line.find("GpuCompiler::OptimizeHloModule fusion pass pipeline for module") != -1:
        out_file.write(line)
      if line.find("GpuCompiler::OptimizeHloModule all reduce combiner pass pipeline for") != -1:
        out_file.write(line)
      
