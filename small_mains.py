import numpy as np
from deploy import call_zac
from classifier import clf_predict_matrix

# Call Zac's 3D CNN model to give us embeddings of the fMRIs, as well as guesses for good or bad
zac_features, zac_scores = call_zac(txt_path='/media/brainhack_data/Track_2/test_txt.txt', model_path='/media/brainhack_data/models/brain/sl_resnet3d/new_backup3/')

# Pass embeddings to pretrained sklearn classifiers
zac_rf_scores = clf_predict_matrix(zac_features, 'clfs/softmax_rf.clf')
zac_knn_scores = clf_predict_matrix(zac_features, 'clfs/softmax_kn.clf')
zac_adaboost_scores = clf_predict_matrix(zac_features, 'clfs/softmax_ada.clf')

# Average all of the results (with a bit of extra weight on Zac's)
final_scores = zac_rf_scores + zac_knn_scores + zac_adaboost_scores + zac_scores + zac_scores + zac_scores

final_scores /= 6.0

# Save the output
np.savetxt('small_scores.txt', np.array(final_scores), newline='\n')
