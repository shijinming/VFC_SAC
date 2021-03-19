import numpy as np

file_sac="../sac/sac.txt"
file_random="../sac/random.txt"
file_greedy="../sac/greedy.txt"
output_file="../sac/plot_tmp.txt"

with open(file_random) as f:
    random_tmp=f.read().split("num_vehicles")[1:]
with open(file_greedy) as f:
    greedy_tmp=f.read().split("num_vehicles")[1:]

random_result=[]
greedy_result=[]
for i in range(10):
    random_item = np.array([0.0]*17)
    greedy_item = np.array([0.0]*17)
    for j in range(100):
        random = random_tmp[j*10+i].split('\n')[2:-1]
        tmp=np.array([0.0]*17)
        utility=-999
        random_result_tmp=""
        for k in random:
            if float(k.split(' ')[0])>utility:
                utility = float(k.split(' ')[0])
                random_result_tmp=k
        random_item+=np.array([float(r) for r in random_result_tmp.split(' ')[:-1]])
        greedy = greedy_tmp[j*10+i].split('\n')[2]
        greedy_item+=np.array([float(g) for g in greedy.split(' ')[:-1]])
    random_result.append(random_item/100)
    greedy_result.append(greedy_item/100)

sac_result=np.array([[0.0]*21]*10)
for i in [1,2,3,5]:
    with open("../sac/sac{}.txt".format(i)) as f:
        sac_tmp=f.read().split("num_vehicles")[1:]
    for j in range(10):
        sac=sac_tmp[j].split('\n')[10:-1]
        utility=-999
        sac_result_tmp=""
        for s in sac:
            if float(s.split(' ')[0])>utility:
                utility = float(s.split(' ')[0])
                sac_result_tmp=s
        tmp=np.array([float(s) for s in sac_result_tmp.split(' ')[:-1]]+[0]*4)
        for n in range(4):
            tmp[n+17]=4-tmp[n+5]//1000
            tmp[n+5]=tmp[n+5]-(4-tmp[n+17])*1000
        sac_result[j]+=tmp
sac_result=np.array([i/4 for i in sac_result])

print(greedy_result)

output=open(output_file,'w+')
output.write("utility_d=[")
output.write(','.join([str(r[0]) for r in random_result])+'\n')
output.write(','.join([str(g[0]) for g in greedy_result])+'\n')
output.write(','.join([str(s[0]) for s in sac_result])+'];\n')
output.write("ratio_h=[")
output.write(','.join([str(sum(r[9:13])/16) for r in random_result])+'\n')
output.write(','.join([str(sum(g[9:13])/16) for g in greedy_result])+'\n')
output.write(','.join([str(sum(s[9:13])/16) for s in sac_result])+'];\n')
output.write("ratio_l=[")
output.write(','.join([str(sum(r[1:5])/16) for r in random_result])+'\n')
output.write(','.join([str(sum(g[1:5])/16) for g in greedy_result])+'\n')
output.write(','.join([str(sum(s[17:21])/16) for s in sac_result])+'];\n')
r=random_result[2]
g=greedy_result[2]
s=sac_result[2]
output.write("delay_h15=[")
output.write(','.join([str(i) for i in r[13:17]/r[9:13]])+'\n')
output.write(','.join([str(i) for i in g[13:17]/g[9:13]])+'\n')
output.write(','.join([str(i) for i in s[13:17]/s[9:13]])+'];\n')
output.write("delay_l15=[")
output.write(','.join([str(i) for i in r[5:9]/r[1:5]])+'\n')
output.write(','.join([str(i) for i in g[5:9]/g[1:5]])+'\n')
output.write(','.join([str(i) for i in s[5:9]/s[17:21]])+'];\n')
r=random_result[6]
g=greedy_result[6]
s=sac_result[6]
output.write("delay_h35=[")
output.write(','.join([str(i) for i in r[13:17]/r[9:13]])+'\n')
output.write(','.join([str(i) for i in g[13:17]/g[9:13]])+'\n')
output.write(','.join([str(i) for i in s[13:17]/s[9:13]])+'];\n')
output.write("delay_l35=[")
output.write(','.join([str(i) for i in r[5:9]/r[1:5]])+'\n')
output.write(','.join([str(i) for i in g[5:9]/g[1:5]])+'\n')
output.write(','.join([str(i) for i in s[5:9]/s[17:21]])+'];\n')

output.close()

# file_random="../sac/random_change.txt"
# file_greedy="../sac/greedy_change.txt"

# with open(file_random) as f:
#     random_tmp=f.read().split("task_num")[1:]
# with open(file_greedy) as f:
#     greedy_tmp=f.read().split("task_num")[1:]

# random_result=[]
# greedy_result=[]
# for i in range(8):
#     random_item = np.array([0.0]*17)
#     greedy_item = np.array([0.0]*17)
#     for j in range(100):
#         random = random_tmp[j*8+i].split('\n')[2:-1]
#         tmp=np.array([0.0]*17)
#         utility=-999
#         random_result_tmp=""
#         for k in random:
#             if float(k.split(' ')[0])>utility:
#                 utility = float(k.split(' ')[0])
#                 random_result_tmp=k
#         random_item+=np.array([float(r) for r in random_result_tmp.split(' ')[:-1]])
#         greedy = greedy_tmp[j*8+i].split('\n')[2]
#         greedy_item+=np.array([float(g) for g in greedy.split(' ')[:-1]])
#     random_result.append(random_item/100/(i+1)/8)
#     greedy_result.append(greedy_item/100/(i+1)/8)

# sac_result=np.array([[0.0]*21]*8)
# for i in range(1,6):
#     with open("../sac_change{}.txt".format(i)) as f:
#         sac_tmp=f.read().split("num_vehicles")[1:]
#     for j in range(8):
#         sac=sac_tmp[j].split('\n')[10:-1]
#         utility=-999
#         sac_result_tmp=""
#         for s in sac:
#             if float(s.split(' ')[0])>utility:
#                 utility = float(s.split(' ')[0])
#                 sac_result_tmp=s
#         tmp=np.array([float(s) for s in sac_result_tmp.split(' ')[:-1]]+[0]*4)
#         tmp[0]=tmp[0]/8/(j+1)
#         sac_result[j]+=tmp
# sac_result=np.array([i/5 for i in sac_result])

# with open(output_file,'a') as output:
#     output.write("utility_t=[\n")
#     output.write(','.join([str(r[0]) for r in random_result])+'\n')
#     output.write(','.join([str(g[0]) for g in greedy_result])+'\n')
#     output.write(','.join([str(s[0]) for s in sac_result])+'];\n')