To run our original model, run the following command:
python train.py

This saves the model parameters to a file called checkpoint.

To obtain our model's predictions of the test data, run the following command:
python test.py

This saves the predictions to preds.pkl.

============

To run our improved model, run the following command:
python train_batchnorm.py

This saves the model parameters to a file called checkpoint.

To obtain our model's predictions of the test data, run the following command:
python test_batchnorm.py

This saves the predictions to preds.pkl.

============

To obtain the evaluation metrics of either model, run: the following command:
python evaluation.py --preds preds.pkl --gts val.pkl

To visualise three predictions from the test set, run the following command:

python visualisation.py --preds preds.pkl --gts val.pkl

This saves the predictions into a file called output_vis.jpg.
