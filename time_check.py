import timeit
import Python_Edu_module as cm

min_ = -100000000000
max_ =  100000000000
NUMBER = 10     # сколько раз повторять в рамках одной серии измерений времени
REPEAT = 10     # сколько раз повторять серия измерений времени


# raw_times1 = timeit.repeat(lambda: cm.g_arr(min_, max_, 1000, 1),
#                           repeat=REPEAT, number=NUMBER)

# raw_times2 = timeit.repeat(lambda: cm.g_set(min_, max_, 1000, 1),
#                           repeat=REPEAT, number=NUMBER)

# raw_times3 = timeit.repeat(lambda: cm.g_ndarr(min_, max_, 1000, 1),
#                           repeat=REPEAT, number=NUMBER)

# print(str(sum(raw_times1)/len(raw_times1) / NUMBER) + ", " + str(sum(raw_times2)/len(raw_times2) / NUMBER) + ", " + str(sum(raw_times3)/len(raw_times3) / NUMBER))



# raw_times1 = timeit.repeat(lambda: cm.g_arr(min_, max_, 10000, 1),
#                           repeat=REPEAT, number=NUMBER)

# raw_times2 = timeit.repeat(lambda: cm.g_set(min_, max_, 10000, 1),
#                           repeat=REPEAT, number=NUMBER)

# raw_times3 = timeit.repeat(lambda: cm.g_ndarr(min_, max_, 10000, 1),
#                           repeat=REPEAT, number=NUMBER)

# print(str(sum(raw_times1)/len(raw_times1) / NUMBER) + ", " + str(sum(raw_times2)/len(raw_times2) / NUMBER) + ", " + str(sum(raw_times3)/len(raw_times3) / NUMBER))


raw_times1 = timeit.repeat(lambda: cm.g_arr(min_, max_, 1000000, 1),
                          repeat=REPEAT, number=NUMBER)

raw_times2 = timeit.repeat(lambda: cm.g_set(min_, max_, 1000000, 1),
                          repeat=REPEAT, number=NUMBER)

raw_times3 = timeit.repeat(lambda: cm.g_ndarr(min_, max_, 1000000, 1),
                          repeat=REPEAT, number=NUMBER)

print(str(sum(raw_times1)/len(raw_times1) / NUMBER) + ", " + str(sum(raw_times2)/len(raw_times2) / NUMBER) + ", " + str(sum(raw_times3)/len(raw_times3) / NUMBER))
