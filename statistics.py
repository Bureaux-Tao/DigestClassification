all = []

with open("./data/all.txt", "r", encoding = 'utf-8') as f:
    for i in f.readlines():
        all.append(i.strip("\n"))

all = list(set(all))

print(len(all))

count_0 = 0
count_1 = 0
count_2 = 0
count_3 = 0
count_4 = 0
count_5 = 0

for i in all:
    if ("胃" not in i) and ("肠" not in i) and ("食管" not in i) and ("肛门" not in i):
        count_0 += 1
        if "\t0" not in i:
            print(i)
        continue
    elif ("胃" in i) and ("肠" in i) and ("食管" in i) and ("肛门" in i):
        count_5 += 1
        continue
    elif "胃" in i:
        count_1 += 1
        continue
    elif "肠" in i:
        count_2 += 1
        continue
    elif "食管" in i:
        count_3 += 1
        continue
    elif "肛门" in i:
        count_4 += 1
        continue

print("都无：", count_0)
print("胃：", count_1)
print("肠：", count_2)
print("食管：", count_3)
print("肛门：", count_4)
print("都有：", count_5)

print(count_0 + count_1 + count_2 + count_3 + count_4 + count_5)

# shuffle(all)
# with open("./data/train_no_dump.txt", "a", encoding = 'utf-8') as f1:
#     with open("./data/val_no_dump.txt", "a", encoding = 'utf-8') as f2:
#         with open("./data/test_no_dump.txt", "a", encoding = 'utf-8') as f3:
#             for i, item in enumerate(all):
#                 if i <= len(all) * 0.8:
#                     f1.write(item + '\n')
#                 elif (i > len(all) * 0.8) and (i <= len(all) * 0.9):
#                     f2.write(item + '\n')
#                 else:
#                     f3.write(item + '\n')
