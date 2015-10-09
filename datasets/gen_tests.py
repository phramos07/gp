FILENAME = "Test_2.txt"

with open(FILENAME, 'wb') as test_:
	for i in range(0, 30):
		test_.write(str(i) + " " + str(i*i*i) + "\n")

