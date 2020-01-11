"""
    file:        parse_output.py

    description: Parse the output logs in order to get the profiling data (time
    that takes to run different models)

    see:
"""

tag_cycles = ' cycles: '
tag_cpu_time = ' cpu_time_used: '

def calculateAvg(lst):
    return sum(lst) / len(lst)

def getDataOfTrace(alltraces, trace2search):
    # search the traces we are interested in
    foundTraces = [ line for line in allTraces if trace2search in line]

    # get information of cycles and cpu time
    foundCycles = [ line for line in foundTraces if tag_cycles in line]
    foundCpuTime = [ line for line in foundTraces if tag_cpu_time in line]

    cyclesValues = []
    cputimeValues = []

    # parse cycles and time and store it in a list
    for run in foundCycles:
        cycles = run.split(tag_cycles)
        cyclesValues.append(float(cycles[1]))

    for run in foundCpuTime:
        cpuTime = run.split(tag_cpu_time)
        cputimeValues.append(float(cpuTime[1]))
    return cyclesValues, cputimeValues

def printData(cycles, times, modelName):
    if cycles != [] and times != []:
        print(""); print("---------------------------")
        print(modelName, "Benchmarking"); print("---------------------------")
        print(modelName, "Averaged values =", len(cycles))
        print(modelName, "Cycles", calculateAvg(cycles))
        print(modelName, "Seconds", calculateAvg(times))
        print(times)

if __name__ == '__main__':
    print("Parsing benchmarking output")

    allTraces = [ line for line in open('benchmarking.txt') if '[benchmark]' in line]

    # Use the tag present in the C code
    mnist_cycles, mnist_times = getDataOfTrace(allTraces, '[mnist]')
    tinyyolov2_cycles, tinyyolov2_times = getDataOfTrace(allTraces, '[tinyyolov2]')

    printData(mnist_cycles, mnist_times, 'mnist')
    printData(tinyyolov2_cycles, tinyyolov2_times, 'tinyyolov2')
