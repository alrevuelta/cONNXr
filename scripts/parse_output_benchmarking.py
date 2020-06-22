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
    foundTraces = [ line for line in alltraces if trace2search in line]

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
        print(modelName, "Averaged runs =", len(cycles))
        print(modelName, "Cycles", calculateAvg(cycles))
        print(modelName, "Seconds", calculateAvg(times))
        print("All times", times)

if __name__ == '__main__':
    print("Parsing benchmarking output")

    models_to_benchmark = ['mnist', 'tinyyolov2', 'super_resolution']
    for model in models_to_benchmark:
        with open(f'benchmarks/{model}.txt') as file:
            lines = file.readlines()
            just_benchmarking = [line for line in lines if '[benchmark]' in line]
            cycles, times = getDataOfTrace(just_benchmarking, f'[{model}]')
            printData(cycles, times, model)
