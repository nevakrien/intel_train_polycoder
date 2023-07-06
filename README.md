KEY ISSUES:

1. metrics are calculated in a non acurate way curently the fact tensors are masked is not really taken into acount
2. the dataset split is diffrent than the paper since we dont have their seed value 
3. multigpu feature in train is curently broken because DATAPARALLEL is not yet implemented by ipex