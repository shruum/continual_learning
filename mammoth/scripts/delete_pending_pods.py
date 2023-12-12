import os
os.system('kubectl -n arl get pods | grep cls-er-tiny > pending_jobs.txt')
lst_jobs = []
count = 0
with open('pending_jobs.txt', 'r') as f:
    for line in f:
        lst_jobs.append(line.split()[0])

for job in lst_jobs:
    count += 1
    os.system('kubectl -n arl delete pod %s' % job)

print(f'{count} jobs deleted')
