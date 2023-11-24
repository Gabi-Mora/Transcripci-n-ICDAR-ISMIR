from editdistance import distance




def SER(y_true, y_pred):

	num = sum([distance([u for u in y_true[it] if u != -1], [u for u in y_pred[it] if u != -1]) for it in range(len(y_true))])
	den = sum([len([u for u in seq if u != -1]) for seq in y_true])

	
	return num/den





if __name__ == '__main__':

	y_true = [
		[1,2,3,4],
		[5,6,7,8]
	]

	y_pred = [
		[1,2,3],
		[5,6,7]
	]

	SER(y_true, y_pred)

	print("hello")