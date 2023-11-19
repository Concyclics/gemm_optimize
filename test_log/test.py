import pandas as pd
import os

data = pd.DataFrame(columns=['size', 'metric', 'count', 'perf', 'unit'])

def parse_output(output, size, data=data):
    lines = output.split('\n')
    for line in lines:
        if '#' in line:
            line = line.split('#')
            line0 = line[0].strip()
            if line0 == '':
                continue
            line1 = line[1].strip()
            print(line0)
            metric = line0.split()[-1]
            count = line0.split()[0]
            line1 = line1.split('(')[0].strip()
            perf = line1.split()[0]
            unit = line1.split()[1:]
            unit = ' '.join(unit)
            data.loc[len(data)] = [size, metric, count, perf, unit]

    return data

# test multiple outputs
codes = ['mat_mul1', 'mat_mul2', 'my_mat_mul']
for code in codes:
    command = 'perf stat -o output -d -d -d -r 10 ' + './' + code
    data = pd.DataFrame(columns=['size', 'metric', 'count', 'perf', 'unit'])
    for size in range(1000, 10001, 1000):
        os.system(command + ' ' + str(size))
        with open('output', 'r') as f:
            output = f.read()
            data = parse_output(output, size, data)
    data.to_csv(code + '.csv', index=False)
    os.system('rm output')
