import os
os.system('kubectl -n arl get jobs | grep cls-er | grep 1/1 > completed_jobs.txt')
# os.system('kubectl -n arl get jobs | grep fahad-joint > completed_jobs.txt')
# os.system('kubectl -n arl get jobs | grep naresh-cs > completed_jobs.txt')
# os.system('kubectl -n arl get jobs | grep fahad > completed_jobs.txt')
# os.system('kubectl -n arl get pods | grep fahad | grep Error > completed_jobs.txt')
lst_jobs = []
count = 0
with open('completed_jobs.txt', 'r') as f:
    for line in f:
        lst_jobs.append(line.split()[0])

for job in lst_jobs:
    count += 1
    os.system('kubectl -n arl delete job %s' % job)

print(f'{count} jobs deleted')
