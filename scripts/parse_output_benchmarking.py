"""
    file:        parse_output.py

    description: Parse the output logs in order to get the profiling data (time
    that takes to run different models)

    see:
"""

def calculateAvg(lst):
    return sum(lst) / len(lst)

if __name__ == '__main__':
    print("Parsing benchmarking output")

    #print("All Traces:")
    allTraces = [ line for line in open('benchmarking.txt') if '[benchmark]' in line]
    #print(allTraces)

    #print("MNIST Traces:")
    mnistTraces = [ line for line in allTraces if '[mnist]' in line]
    #print(mnistTraces)

    # Take execution time and cycles
    # Tags are:
    tag_cycles = ' cycles: '
    tag_cpu_time = ' cpu_time_used: '

    mnistCycles = [ line for line in mnistTraces if tag_cycles in line]
    cyclesValues = []
    for run in mnistCycles:
        cycles = run.split(tag_cycles)
        cyclesValues.append(float(cycles[1]))

    mnistCpuTime = [ line for line in mnistTraces if tag_cpu_time in line]
    cputimeValues = []
    for run in mnistCpuTime:
        mnistCpuTime = run.split(tag_cpu_time)
        cputimeValues.append(float(mnistCpuTime[1]))

    print(""); print("---------------------------")
    print("MNIST Benchmarking"); print("---------------------------")
    print("MNIST Averaged values =", len(cyclesValues))
    print("MNIST Cycles", calculateAvg(cyclesValues))
    print("MNIST Seconds", calculateAvg(cputimeValues))
    print(cputimeValues)
