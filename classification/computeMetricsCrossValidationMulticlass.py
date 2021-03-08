from copy import deepcopy

ZERO = "0"
ONE = "1"
TWO = "2"
THREE = "3"

factual_social = [
	["Fold 0tp: [0 2 1 0]tn: [4 2 5 6]fp: [0 3 0 1]fn: [3 0 1 0]",
	[0, 1, 2, 3], [3, 2, 2, 0]],
	["Fold 1tp: [0 0 0 0]tn: [3 0 5 6]fp: [0 6 0 1]fn: [4 1 2 0]",
	[0, 1, 2, 3], [4, 1, 2, 0]],
	["Fold 2tp: [0 1 1 0]tn: [4 5 2 5]fp: [0 1 2 2]fn: [3 0 2 0]",
	[0, 1, 2, 3], [3, 1, 3, 0]],
	["Fold 3tp: [0 2 0 1]tn: [3 2 6 6]fp: [0 3 1 0]fn: [4 0 0 0]",
	[0, 1, 2, 3], [4, 2, 0, 1]],
	["Fold 4tp: [0 1 2]tn: [4 2 3]fp: [0 3 0]fn: [2 0 1]",
	[0, 1, 3], [2, 1, 3]],
	["Fold 5tp: [1 1 0]tn: [2 2 4]fp: [0 3 1]fn: [3 0 1]",
	[0, 1, 3], [4, 1, 1]],
	["Fold 6tp: [0 1 0 0]tn: [5 1 4 3]fp: [0 4 0 1]fn: [1 0 2 2]",
	[0, 1, 2, 3], [1, 1, 2, 2]],
	["Fold 7tp: [0 1 0 1]tn: [5 2 4 3]fp: [0 2 1 1]fn: [1 1 1 1]",
	[0, 1, 2, 3], [1, 2, 1, 2]],
	["Fold 8tp: [2 0]tn: [0 2]fp: [4 0]fn: [0 4]",
	[1, 3], [2, 4]],
	["Fold 9tp: [0 0 0 0]tn: [4 1 4 3]fp: [1 3 1 1]fn: [1 2 1 2]",
	[0, 1, 2, 3], [1, 2, 1, 2]]
]

factual_justitie = [
	["Fold 0tp: [0 1 0 0]tn: [4 3 8 6]fp: [0 5 2 2]fn: [6 1 0 2]",
	[0, 1, 2, 3], [6, 2, 0, 2]],
	["Fold 1tp: [0 1 0 2]tn: [8 7 6 2]fp: [0 1 3 3]fn: [2 1 1 3]",
	[0, 1, 2, 3], [2, 2, 1, 5]],
	["Fold 2tp: [0 1 0 5]tn: [8 9 8 1]fp: [0 0 0 4]fn: [2 0 2 0]",
	[0, 1, 2, 3], [2, 1, 2, 5]],
	["Fold 3tp: [0 0 0 2]tn: [9 7 5 1]fp: [0 0 2 6]fn: [1 3 3 1]",
	[0, 1, 2, 3], [1, 3, 3, 3]],
	["Fold 4tp: [0 1 6]tn: [9 7 1]fp: [1 1 1]fn: [0 1 2]",
	[1, 2, 3], [0, 2, 8]],
	["Fold 5tp: [0 0 0 4]tn: [8 7 6 3]fp: [1 3 2 0]fn: [1 0 2 3]",
	[0, 1, 2, 3], [1, 0, 2, 7]],
	["Fold 6tp: [0 0 1 3]tn: [8 6 7 3]fp: [0 3 1 2]fn: [2 1 1 2]",
	[0, 1, 2, 3], [2, 1, 2, 5]],
	["Fold 7tp: [0 0 0 6]tn: [8 9 9 0]fp: [1 1 0 2]fn: [1 0 1 2]",
	[0, 1, 2, 3], [1, 0, 1, 8]]
]

factual_coronavirus = [
	["Fold 0tp: [0 1 0 0]tn: [5 0 6 4]fp: [0 5 0 1]fn: [2 1 1 2]",
	[0, 1, 2, 3], [2, 2, 1, 2]],
	["Fold 1tp: [0 0 1]tn: [3 4 1]fp: [2 1 3]fn: [2 2 2]",
	[1, 2, 3], [2, 2, 3]],
	["Fold 2tp: [0 0 0 0]tn: [2 3 6 3]fp: [0 4 1 2]fn: [5 0 0 2]",
	[0, 1, 2, 3], [5, 0, 0, 2]],
	["Fold 3tp: [0 0 1 1]tn: [4 6 3 3]fp: [0 1 2 2]fn: [3 0 1 1]",
	[0, 1, 2, 3], [3, 0, 2, 2]],
	["Fold 4tp: [0 0 1 1]tn: [4 3 6 3]fp: [0 4 0 1]fn: [3 0 0 2]",
	[0, 1, 2, 3], [3, 0, 1, 3]],
	["Fold 5tp: [0 1 0 1]tn: [5 3 3 5]fp: [0 2 3 0]fn: [2 1 1 1]",
	[0, 1, 2, 3], [2, 2, 1, 2]],
	["Fold 6tp: [0 1 1 0]tn: [5 3 5 3]fp: [0 3 0 2]fn: [2 0 1 2]",
	[0, 1, 2, 3], [2, 1, 2, 2]],
	["Fold 7tp: [0 1 0 1]tn: [4 2 4 6]fp: [1 3 1 0]fn: [2 1 2 0]",
	[0, 1, 2, 3], [2, 2, 2, 1]],
	["Fold 8tp: [0 1 1 1]tn: [6 2 3 6]fp: [1 2 1 0]fn: [0 2 2 0]",
	[0, 1, 2, 3], [0, 3, 3, 1]],
	["Fold 9tp: [1 1 0 0]tn: [5 5 4 2]fp: [0 1 2 2]fn: [1 0 1 3]",
	[0, 1, 2, 3], [2, 1, 1, 3]]
]

factual_externe = [
	["Fold 0tp: [1 1 0 1]tn: [4 3 6 4]fp: [0 2 1 1]fn: [2 1 0 1]",
	[0, 1, 2, 3], [3, 2, 0, 2]],
	["Fold 1tp: [0 0 0 0]tn: [4 2 4 4]fp: [1 5 1 0]fn: [2 0 2 3]",
	[0, 1, 2, 3], [2, 0, 2, 3]],
	["Fold 2tp: [0 0 0 1]tn: [3 3 4 3]fp: [0 3 1 1]fn: [3 0 1 1]",
	[0, 1, 2, 3], [3, 0, 1, 2]],
	["Fold 3tp: [0 1 1 0]tn: [3 3 3 5]fp: [0 2 1 1]fn: [3 0 1 0]",
	[0, 1, 2, 3], [3, 1, 2, 0]],
	["Fold 4tp: [1 0 0 1]tn: [1 4 5 4]fp: [1 2 0 1]fn: [3 0 1 0]",
	[0, 1, 2, 3], [4, 0, 1, 1]],
	["Fold 5tp: [1 0 1]tn: [1 5 2]fp: [3 1 0]fn: [1 0 3]",
	[1, 2, 3], [2, 0, 4]],
	["Fold 6tp: [2 0 1 2]tn: [3 5 5 4]fp: [0 1 0 0]fn: [1 0 0 0]",
	[0, 1, 2, 3], [3, 0, 1, 2]],
	["Fold 7tp: [0 1 1]tn: [3 3 2]fp: [0 2 2]fn: [3 0 1]",
	[0, 1, 3], [3, 1, 2]],
	["Fold 8tp: [0 0 1 0]tn: [3 3 3 4]fp: [1 3 1 0]fn: [2 0 1 2]",
	[0, 1, 2, 3], [2, 0, 2, 2]],
	["Fold 9tp: [1 1 0]tn: [3 1 4]fp: [0 3 1]fn: [2 1 1]",
	[0, 1, 3], [3, 2, 1]]
]

factual_sanatate_mediu = [
	["Fold 0tp: [0 1 0]tn: [4 0 3]fp: [0 5 0]fn: [2 0 3]",
	[0, 1, 3], [2, 1, 3]],
	["Fold 1tp: [0 0 0 1]tn: [4 5 3 1]fp: [0 1 2 2]fn: [2 0 1 2]",
	[0, 1, 2, 3], [2, 0, 1, 3]],
	["Fold 2tp: [1 0 0]tn: [2 3 2]fp: [2 0 3]fn: [1 3 1]",
	[1, 2, 3], [2, 3, 1]],
	["Fold 3tp: [0 0 0]tn: [1 1 3]fp: [0 4 1]fn: [4 0 1]",
	[0, 1, 3], [4, 0, 1]],
	["Fold 4tp: [0 0 0 1]tn: [2 1 4 4]fp: [0 4 0 0]fn: [3 0 1 0]",
	[0, 1, 2, 3], [3, 0, 1, 1]],
	["Fold 5tp: [1 0 0 1]tn: [3 4 3 2]fp: [0 1 0 2]fn: [1 0 2 0]",
	[0, 1, 2, 3], [2, 0, 2, 1]],
	["Fold 6tp: [0 1 0 0]tn: [4 1 3 3]fp: [0 2 0 2]fn: [1 1 2 0]",
	[0, 1, 2, 3], [1, 2, 2, 0]],
	["Fold 7tp: [0 0 1 0]tn: [4 4 1 2]fp: [0 1 1 2]fn: [1 0 2 1]",
	[0, 1, 2, 3], [1, 0, 3, 1]],
	["Fold 8tp: [0 0 0 1]tn: [3 2 3 3]fp: [1 3 0 0]fn: [1 0 2 1]",
	[0, 1, 2, 3], [1, 0, 2, 2]],
	["Fold 9tp: [1 0 0]tn: [1 3 2]fp: [1 2 1]fn: [2 0 2]",
	[1, 2, 3], [3, 0, 2]]
]

factual_politica = [
	["Fold 0tp: [3 1 1 5]tn: [15 21 21 11]fp: [3 4 5 7]fn: [8 3 2 6]",
	[0, 1, 2, 3], [11, 4, 3, 11]],
	["Fold 1tp: [2 1 0 6]tn: [15 19 22 11]fp: [5 4 2 9]fn: [7 5 5 3]",
	[0, 1, 2, 3], [9, 6, 5, 9]],
	["Fold 2tp: [0 1 1 1]tn: [16 15 21 9]fp: [5 9 3 9]fn: [8 4 4 10]",
	[0, 1, 2, 3], [8, 5, 5, 11]],
	["Fold 3tp: [3 4 0 3]tn: [20 17 18 13]fp: [2 5 1 11]fn: [4 3 10 2]",
	[0, 1, 2, 3], [7, 7, 10, 5]],
	["Fold 4tp: [0 0 0 6]tn: [16 21 20 5]fp: [7 5 5 5]fn: [5 2 3 12]",
	[0, 1, 2, 3], [5, 2, 3, 18]],
	["Fold 5tp: [3 0 0 11]tn: [17 24 21 8]fp: [5 4 3 2]fn: [3 0 4 7]",
	[0, 1, 2, 3], [6, 0, 4, 18]],
	["Fold 6tp: [2 0 1 13]tn: [18 27 22 5]fp: [6 1 3 2]fn: [2 0 2 8]",
	[0, 1, 2, 3], [4, 0, 3, 21]],
	["Fold 7tp: [4 4 1 3]tn: [14 20 19 15]fp: [3 1 4 8]fn: [7 3 4 2]",
	[0, 1, 2, 3], [11, 7, 5, 5]],
	["Fold 8tp: [4 1 0 3]tn: [12 21 22 9]fp: [7 4 1 8]fn: [5 2 5 8]",
	[0, 1, 2, 3], [9, 3, 5, 11]],
	["Fold 9tp: [1 1 0 4]tn: [18 17 22 5]fp: [5 3 1 13]fn: [4 7 5 6]",
	[0, 1, 2, 3], [5, 8, 5, 10]]
]

factual_economie = [
	["Fold 0tp: [1 1 2 2]tn: [14 16 17 17]fp: [3 8 6 6]fn: [11 4 4 4]",
	[0, 1, 2, 3], [12, 5, 6, 6]],
	["Fold 1tp: [0 3 1 1]tn: [19 12 16 16]fp: [0 13 4 7]fn: [10 1 8 5]",
	[0, 1, 2, 3], [10, 4, 9, 6]],
	["Fold 2tp: [4 1 0 2]tn: [6 19 22 18]fp: [12 6 0 4]fn: [7 3 7 5]",
	[0, 1, 2, 3], [11, 4, 7, 7]],
	["Fold 3tp: [1 3 2 1]tn: [11 20 19 15]fp: [15 0 1 6]fn: [2 6 7 7]",
	[0, 1, 2, 3], [3, 9, 9, 8]],
	["Fold 4tp: [10 2 0 1]tn: [8 17 24 20]fp: [9 4 1 1]fn: [1 5 3 6]",
	[0, 1, 2, 3], [11, 7, 3, 7]],
	["Fold 5tp: [8 2 0 1]tn: [7 19 19 22]fp: [9 2 3 3]fn: [4 5 6 2]",
	[0, 1, 2, 3], [12, 7, 6, 3]],
	["Fold 6tp: [3 4 0 1]tn: [16 11 22 15]fp: [4 8 0 8]fn: [5 5 6 4]",
	[0, 1, 2, 3], [8, 9, 6, 5]],
	["Fold 7tp: [3 0 2 2]tn: [12 17 18 16]fp: [8 7 1 5]fn: [5 4 7 5]",
	[0, 1, 2, 3], [8, 4, 9, 7]],
	["Fold 8tp: [5 0 0 4]tn: [14 20 21 10]fp: [5 8 3 3]fn: [4 0 4 11]",
	[0, 1, 2, 3], [9, 0, 4, 15]],
	["Fold 9tp: [6 1 1 1]tn: [10 20 20 15]fp: [7 3 5 4]fn: [5 4 2 8]",
	[0, 1, 2, 3], [11, 5, 3, 9]]
]

factual_texte_mici = [
	["Fold 0tp: [5 3 2 6]tn: [26 28 38 26]fp: [8 12 8 7]fn: [12 8 3 12]",
	[0, 1, 2, 3], [17, 11, 5, 18]],
	["Fold 1tp: [6 1 1 10]tn: [24 38 36 22]fp: [7 8 6 12]fn: [14 4 8 7]",
	[0, 1, 2, 3], [20, 5, 9, 17]],
	["Fold 2tp: [6 0 4 5]tn: [21 36 30 30]fp: [16 6 3 11]fn: [8 9 14 5]",
	[0, 1, 2, 3], [14, 9, 18, 10]],
	["Fold 3tp: [6 4 0 5]tn: [21 33 40 23]fp: [11 8 6 11]fn: [13 6 5 12]",
	[0, 1, 2, 3], [19, 10, 5, 17]],
	["Fold 4tp: [9 2 1 3]tn: [19 31 36 31]fp: [13 10 8 5]fn: [10 8 6 12]",
	[0, 1, 2, 3], [19, 10, 7, 15]],
	["Fold 5tp: [6 2 0 10]tn: [26 35 40 19]fp: [11 8 2 12]fn: [8 6 9 10]",
	[0, 1, 2, 3], [14, 8, 9, 20]],
	["Fold 6tp: [3 0 1 11]tn: [31 37 31 16]fp: [9 11 5 10]fn: [7 2 13 13]",
	[0, 1, 2, 3], [10, 2, 14, 24]],
	["Fold 7tp: [9 1 1 15]tn: [24 43 42 17]fp: [11 4 6 3]fn: [6 2 1 15]",
	[0, 1, 2, 3], [15, 2, 3, 30]],
	["Fold 8tp: [7 1 1 8]tn: [17 40 40 20]fp: [16 0 3 14]fn: [10 9 6 8]",
	[0, 1, 2, 3], [17, 10, 7, 16]],
	["Fold 9tp: [4 0 1 12]tn: [27 33 38 19]fp: [12 5 3 13]fn: [7 12 8 6]",
	[0, 1, 2, 3], [11, 12, 9, 18]]
]

factual_texte_mari = [
	["Fold 0tp: [4 0 1 3]tn: [16 20 28 26]fp: [7 15 3 8]fn: [14 6 9 4]",
	[0, 1, 2, 3], [18, 6, 10, 7]],
	["Fold 1tp: [5 0 4 0]tn: [15 32 20 24]fp: [8 3 11 10]fn: [13 6 6 7]",
	[0, 1, 2, 3], [18, 6, 10, 7]],
	["Fold 2tp: [5 2 2 5]tn: [20 29 27 20]fp: [11 4 4 8]fn: [5 6 8 8]",
	[0, 1, 2, 3], [10, 8, 10, 13]],
	["Fold 3tp: [9 2 2 1]tn: [13 28 32 23]fp: [14 2 3 8]fn: [5 9 4 9]",
	[0, 1, 2, 3], [14, 11, 6, 10]],
	["Fold 4tp: [5 3 2 2]tn: [18 27 22 27]fp: [9 4 8 8]fn: [9 7 9 4]",
	[0, 1, 2, 3], [14, 11, 10, 6]],
	["Fold 5tp: [4 4 2 8]tn: [22 29 27 22]fp: [13 2 2 6]fn: [2 6 10 5]",
	[0, 1, 2, 3], [6, 10, 12, 13]],
	["Fold 6tp: [7 1 2 3]tn: [17 32 28 18]fp: [14 4 5 5]fn: [3 4 6 15]",
	[0, 1, 2, 3], [10, 5, 8, 18]],
	["Fold 7tp: [5 1 3 2]tn: [18 34 23 18]fp: [10 1 12 7]fn: [8 5 3 14]",
	[0, 1, 2, 3], [13, 6, 6, 16]],
	["Fold 8tp: [2 0 1 9]tn: [29 34 20 11]fp: [7 4 13 5]fn: [3 3 7 16]",
	[0, 1, 2, 3], [5, 3, 8, 25]],
	["Fold 9tp: [5 0 0 6]tn: [18 31 29 15]fp: [13 4 3 10]fn: [5 6 9 10]",
	[0, 1, 2, 3], [10, 6, 9, 16]]
]

factual_fara_nume = [
	["Fold 0tp: [0 2 0 5]tn: [20 9 28 12]fp: [0 18 1 5]fn: [11 2 2 9]",
	[0, 1, 2, 3], [11, 4, 2, 14]],
	["Fold 1tp: [2 2 2 2]tn: [17 19 19 15]fp: [5 7 2 9]fn: [7 3 8 5]",
	[0, 1, 2, 3], [9, 5, 10, 7]],
	["Fold 2tp: [2 1 0 3]tn: [16 20 21 11]fp: [8 4 4 9]fn: [5 6 6 8]",
	[0, 1, 2, 3], [7, 7, 6, 11]],
	["Fold 3tp: [5 4 1 0]tn: [16 14 24 18]fp: [3 11 3 4]fn: [7 2 3 9]",
	[0, 1, 2, 3], [12, 6, 4, 9]],
	["Fold 4tp: [3 4 4 3]tn: [21 18 18 19]fp: [4 6 2 5]fn: [3 3 7 4]",
	[0, 1, 2, 3], [6, 7, 11, 7]],
	["Fold 5tp: [0 1 1 10]tn: [25 21 19 7]fp: [0 7 2 9]fn: [5 1 8 4]",
	[0, 1, 2, 3], [5, 2, 9, 14]],
	["Fold 6tp: [2 2 1 7]tn: [21 15 25 11]fp: [1 12 1 4]fn: [6 1 3 8]",
	[0, 1, 2, 3], [8, 3, 4, 15]],
	["Fold 7tp: [4 0 0 7]tn: [13 26 20 12]fp: [7 2 8 2]fn: [6 2 2 9]",
	[0, 1, 2, 3], [10, 2, 2, 16]],
	["Fold 8tp: [2 0 0 7]tn: [15 24 25 5]fp: [6 1 3 11]fn: [7 5 2 7]",
	[0, 1, 2, 3], [9, 5, 2, 14]],
	["Fold 9tp: [2 0 1 7]tn: [21 19 20 10]fp: [3 3 4 10]fn: [4 8 5 3]",
	[0, 1, 2, 3], [6, 8, 6, 10]]
]

factual_cu_nume = [
	["Fold 0tp: [14 1 5 5]tn: [29 41 40 39]fp: [10 9 10 8]fn: [9 11 7 10]",
	[0, 1, 2, 3], [23, 12, 12, 15]],
	["Fold 1tp: [12 0 1 5]tn: [18 50 34 38]fp: [18 2 13 10]fn: [13 9 13 8]",
	[0, 1, 2, 3], [25, 9, 14, 13]],
	["Fold 2tp: [9 1 5 8]tn: [28 42 42 33]fp: [12 6 7 13]fn: [12 12 7 7]",
	[0, 1, 2, 3], [21, 13, 12, 15]],
	["Fold 3tp: [8 3 3 4]tn: [24 48 37 31]fp: [18 3 8 14]fn: [11 7 13 12]",
	[0, 1, 2, 3], [19, 10, 16, 16]],
	["Fold 4tp: [15 3 2 3]tn: [15 45 46 39]fp: [18 4 9 7]fn: [13 9 4 12]",
	[0, 1, 2, 3], [28, 12, 6, 15]],
	["Fold 5tp: [6 2 3 5]tn: [28 44 39 27]fp: [16 3 8 18]fn: [11 12 11 11]",
	[0, 1, 2, 3], [17, 14, 14, 16]],
	["Fold 6tp: [3 1 2 12]tn: [33 50 37 20]fp: [15 2 13 13]fn: [10 8 9 16]",
	[0, 1, 2, 3], [13, 9, 11, 28]],
	["Fold 7tp: [4 1 4 8]tn: [32 50 30 27]fp: [15 5 19 5]fn: [10 5 8 21]",
	[0, 1, 2, 3], [14, 6, 12, 29]],
	["Fold 8tp: [7 0 3 19]tn: [34 52 45 20]fp: [14 4 6 8]fn: [6 5 7 14]",
	[0, 1, 2, 3], [13, 5, 10, 33]],
	["Fold 9tp: [7 1 3 7]tn: [23 47 46 24]fp: [20 2 3 18]fn: [11 11 9 12]",
	[0, 1, 2, 3], [18, 12, 12, 19]]
]

datasets = {
	"factual_social": [factual_social, "Category: Social"],
	"factual_economie": [factual_economie, "Category: Economics"],
	"factual_politica": [factual_politica, "Category: Politcs"],
	"factual_externe": [factual_externe, "Category: International"],
	"factual_coronavirus": [factual_coronavirus, "Category: Coronavirus"],
	"factual_justitie": [factual_justitie, "Category: Justice"],
	"factual_sanatate_mediu": [factual_sanatate_mediu, "Category: Health"],
	"factual_texte_mici": [factual_texte_mici, "Short statements"],
	"factual_texte_mari": [factual_texte_mari, "Long statements"],
	"factual_cu_nume": [factual_cu_nume, "Statements containing at least one name of person/organization"],
	"factual_fara_nume": [factual_fara_nume, "Statements not containing any name of person/organization" ]
}

def convert_label_int_to_str(label):
	if label == 0:
		return ZERO
	elif label == 1:
		return ONE
	elif label == 2:
		return TWO
	elif label == 3:
		return THREE

def init_data():
	d = {}
	d[ZERO] = 0
	d[ONE] = 0
	d[TWO] = 0
	d[THREE] = 0
	return d

def parse_elem(data_list):
	tp, tn, fp, fn = init_data(), init_data(), init_data(), init_data()
	support = init_data()

	for data in data_list:
		elem, labels, no = data
		no = deepcopy(no)
		tokens = elem.split(":")
		tp_list = [int(x) for x in tokens[1][2:-3].split(" ")]
		tn_list = [int(x) for x in tokens[2][2:-3].split(" ")]
		fp_list = [int(x) for x in tokens[3][2:-3].split(" ")]
		fn_list = [int(x) for x in tokens[4][2:-1].split(" ")]
		for label in labels:
			str_label = convert_label_int_to_str(label)
			tp_ = tp_list.pop(0)
			tn_ = tn_list.pop(0)
			fp_ = fp_list.pop(0)
			fn_ = fn_list.pop(0)
			tp[str_label] += tp_
			tn[str_label] += tn_
			fp[str_label] += fp_
			fn[str_label] += fn_
			no_ = no.pop(0)
			support[str_label] += no_

	return tp, tn, fp, fn, support

def parse_(data):
	tp, tn, fp, fn = init_data(), init_data(), init_data(), init_data()
	support = init_data()
	elem, labels, no = data
	tokens = elem.split(":")
	tp_list = [int(x) for x in tokens[1][2:-3].split(" ")]
	tn_list = [int(x) for x in tokens[2][2:-3].split(" ")]
	fp_list = [int(x) for x in tokens[3][2:-3].split(" ")]
	fn_list = [int(x) for x in tokens[4][2:-1].split(" ")]
	for label in labels:
		str_label = convert_label_int_to_str(label)
		tp_ = tp_list.pop(0)
		tn_ = tn_list.pop(0)
		fp_ = fp_list.pop(0)
		fn_ = fn_list.pop(0)
		tp[str_label] += tp_
		tn[str_label] += tn_
		fp[str_label] += fp_
		fn[str_label] += fn_
		no_ = no.pop(0)
		support[str_label] += no_

	return tp, tn, fp, fn, support

def compute_scores(tp, tn, fp, fn, support, no_classes):
	precision = init_data()
	recall = init_data()
	f1_score = init_data()

	for key in precision.keys():
		if tp[key] + fp[key] != 0:
			precision[key] = tp[key] / (tp[key] + fp[key])
		else:
			precision[key] = 0
		if tp[key] + fn[key] != 0:
			recall[key] = tp[key] / (tp[key] + fn[key])
		else:
			recall[key] = 0
		if precision[key] + recall[key] != 0:
			f1_score[key] = 2 * precision[key] * recall[key] / (precision[key] + recall[key])
		else:
			f1_score[key] = 0

	macro_avg = {}
	weighted_avg = {}

	no = len(no_classes)
	macro_avg["precision"] = sum([precision[key] for key in precision.keys()]) / no
	macro_avg["recall"] = sum([recall[key] for key in precision.keys()]) / no
	macro_avg["f1_score"] = sum([f1_score[key] for key in precision.keys()]) / no

	total = sum([support[key] for key in precision.keys()])
	weighted_avg["precision"] = sum([precision[key] * support[key] for key in precision.keys()]) / total
	weighted_avg["recall"] = sum([recall[key] * support[key] for key in precision.keys()]) / total
	weighted_avg["f1_score"] = sum([f1_score[key] * support[key] for key in precision.keys()]) / total

	accuracy = sum([tp[key] for key in precision.keys()]) / total

	return accuracy, macro_avg, weighted_avg, precision, recall, f1_score

def run_statistics(dataset):
	tp, tn, fp, fn, support = parse_elem(dataset)
	accuracy, macro_avg, weighted_avg, precision, recall, f1_score = \
		compute_scores(tp, tn, fp, fn, support, [0, 1, 2, 3])

	tp_list = []
	tn_list = []
	fp_list = []
	fn_list = []
	for key in tp.keys():
		tp_list.append(tp[key])
		tn_list.append(tn[key])
		fp_list.append(fp[key])
		fn_list.append(fn[key])

	# print ("tp:", tp_list)
	# print ("tn:", tn_list)
	# print ("fp:", fp_list)
	# print ("fn:", fn_list)

	print ("            precision   recall   f1_score   support")
	for key in tp.keys():
		print("" + key + "               " + str(format(round(precision[key], 3), ".3f")) + "    " + \
			str(format(round(recall[key], 3), ".3f")) + "      " + \
			str(format(round(f1_score[key], 3), ".3f")) + "      " +
			str(support[key]) + " ")

	print ("Accuracy:", accuracy)
	print ("macro avg:     ", str(format(round(macro_avg['precision'], 3), ".3f")), "  ",
						 str(format(round(macro_avg['recall'], 3), ".3f")), "    ",
						 str(format(round(macro_avg['f1_score'], 3), ".3f")))
	print ("weighted avg:  ", str(format(round(weighted_avg['precision'], 3), ".3f")), "  ",
						 str(format(round(weighted_avg['recall'], 3), ".3f")), "    ",
						 str(format(round(weighted_avg['f1_score'], 3), ".3f")))

	print ("")
	print ("TN TP FN FP total method")
	print ("Accuracy:", accuracy)
	print ("macro avg:", macro_avg)
	print ("weighted avg:", weighted_avg)

	accuracy_list = []
	macro_avg_list = []
	weighted_avg_list = []
	for elem in dataset:
		tp, tn, fp, fn, support = parse_(elem)
		accuracy, macro_avg, weighted_avg, _, _, _ = compute_scores(tp, tn, fp, fn, support, elem[1])
		accuracy_list.append(accuracy)
		macro_avg_list.append(macro_avg)
		weighted_avg_list.append(weighted_avg)

	print ("=============")
	print ("MACRO/WEIGHTED/ACC Average method")
	print ("Accuracy:", sum(accuracy_list) / 10)

	new_macro_avg = {}
	new_weighted_avg = {}
	keys_list = ['precision', 'recall', 'f1_score']
	for key in keys_list:
		new_macro_avg[key] = sum([x[key] for x in macro_avg_list]) / 10
		new_weighted_avg[key] = sum([x[key] for x in weighted_avg_list]) / 10

	print ("macro avg:", new_macro_avg)
	print ("weighted avg:", new_weighted_avg)

def main():
	for key in datasets.keys():
		dataset = datasets[key]
		print ("")
		print (" ======== ", dataset[1], " ======== ")
		run_statistics(dataset[0])

if __name__ == main():
	main()
