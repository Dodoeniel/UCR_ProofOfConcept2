import csv
import numpy as np


test = ['Hallo1', 'Hallo2']
Wert = np.zeros(2)
Wert[0] = 1
Wert[1] = 2

with open('Ergebnis.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
	for count in range(0,2):
		writer.writerow([test[count] +','+ np.array2string(Wert[count])])



