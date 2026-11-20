import timeit
import Python_Edu_module as cm

min_ = -100000000000
max_ =  100000000000
NUMBER = 10     # сколько раз повторять в рамках одной серии измерений времени
REPEAT = 10     # сколько раз повторять серия измерений времени

# raw_times2 = timeit.repeat(lambda: cm.g_set(min_, max_, 1000, 1),
#                             repeat=REPEAT, number=NUMBER)

# raw_times3 = timeit.repeat(lambda: cm.g_ndarr(min_, max_, 1000, 1),
#                             repeat=REPEAT, number=NUMBER)

# raw_times4 = timeit.repeat(lambda: cm.g_ndarr_fast(min_, max_, 1000, 1),
#                             repeat=REPEAT, number=NUMBER)
# print(str(sum(raw_times2)/len(raw_times2) / NUMBER) + ", " + str(sum(raw_times3)/len(raw_times3) / NUMBER) + ", " + str(sum(raw_times4)/len(raw_times4) / NUMBER))


# for i in range(2500, 10001, 2500):
#     raw_times2 = timeit.repeat(lambda: cm.g_set(min_, max_, i, 1),
#                             repeat=REPEAT, number=NUMBER)

#     raw_times3 = timeit.repeat(lambda: cm.g_ndarr(min_, max_, i, 1),
#                             repeat=REPEAT, number=NUMBER)

#     raw_times4 = timeit.repeat(lambda: cm.g_ndarr_fast(min_, max_, i, 1),
#                             repeat=REPEAT, number=NUMBER)

#     print("цикл " + str(i) + ": " + str(sum(raw_times2)/len(raw_times2) / NUMBER) + ", " + str(sum(raw_times3)/len(raw_times3) / NUMBER) + ", " + str(sum(raw_times4)/len(raw_times4) / NUMBER))


# raw_times2 = timeit.repeat(lambda: cm.g_set(min_, max_, 50000, 1),
#                             repeat=REPEAT, number=NUMBER)

# raw_times3 = timeit.repeat(lambda: cm.g_ndarr(min_, max_, 50000, 1),
#                             repeat=REPEAT, number=NUMBER)

# raw_times4 = timeit.repeat(lambda: cm.g_ndarr_fast(min_, max_, 50000, 1),
#                             repeat=REPEAT, number=NUMBER)
# print(str(sum(raw_times2)/len(raw_times2) / NUMBER) + ", " + str(sum(raw_times3)/len(raw_times3) / NUMBER) + ", " + str(sum(raw_times4)/len(raw_times4) / NUMBER))


# raw_times2 = timeit.repeat(lambda: cm.g_set(min_, max_, 100000, 1),
#                             repeat=REPEAT, number=NUMBER)

# raw_times3 = timeit.repeat(lambda: cm.g_ndarr(min_, max_, 100000, 1),
#                             repeat=REPEAT, number=NUMBER)

# raw_times4 = timeit.repeat(lambda: cm.g_ndarr_fast(min_, max_, 100000, 1),
#                             repeat=REPEAT, number=NUMBER)
# print(str(sum(raw_times2)/len(raw_times2) / NUMBER) + ", " + str(sum(raw_times3)/len(raw_times3) / NUMBER) + ", " + str(sum(raw_times4)/len(raw_times4) / NUMBER))



# # print(cm.g_ndarr_fast(0, 100, 50, 1))


# # for i in range(19_900_000, 20_000_000+1, 100_000):
# #     # время
# #     times = timeit.repeat(lambda: cm.g_ndarr_fast(min_, max_, i, 1),
# #                           repeat=REPEAT, number=NUMBER)
# #     avg_time = sum(times) / len(times) / NUMBER

# #     # память
# #     tracemalloc.start()
# #     cm.g_ndarr_fast(min_, max_, i, 1)
# #     _, peak = tracemalloc.get_traced_memory()
# #     tracemalloc.stop()

# #     print(f"{i}, время: {avg_time:.8f} с, пик памяти: {peak/1024/1024:.2f} МБ")

print(cm.g_ndarr_fast(-100000000000, 100000000000, 10, 1))