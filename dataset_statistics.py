from utils import readFormattedFile, dataset_statistics

tokens, labels,unique = readFormattedFile("n2c2_2014/training3.tsv",mapping="n2c2")
#t2, l2, uL = readFormattedFile("n2c2_2014/training2.tsv",mapping="n2c2_removeBIO")
#tokens.extend(t2)
#labels.extend(l2)
print(len(tokens))
dataset_statistics(tokens,labels,unique)