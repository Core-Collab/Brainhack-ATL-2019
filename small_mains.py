# from siamese_embed_features import call_andrew
import numpy as np
from deploy import call_zac
# from generate_features import call_manisha
from classifier import clf_predict_matrix

# andrew_features = call_andrew(data_path='/media/data/Track_2/test_txt.txt', model_path='./model_checkpoints/siamese_final.pth.tar', use_gpu=True)

# manisha_features = call_manisha(data_path='/media/data/Track_2/test_txt.txt', model_path='/nethome/mnatarajan30/codes/brain_code/log/models/autoencoder/')

zac_features, zac_scores = call_zac(txt_path='/media/data/Track_2/test_txt.txt', model_path='/media/data/models/brain/sl_resnet3d/new_backup3/')

# andrew_rf_scores = clf_predict_matrix(andrew_features, 'clfs/siamese_rf.clf')
# # andrew_boost_scores = clf_predict_matrix(andrew_features, 'clfs/siamese_grad.clf')
# andrew_knn_scores = clf_predict_matrix(andrew_features, 'clfs/siamese_kn.clf')
# andrew_adaboost_scores = clf_predict_matrix(andrew_features, 'clfs/siamese_ada.clf')

zac_rf_scores = clf_predict_matrix(zac_features, 'clfs/softmax_rf.clf')
# zac_boost_scores = clf_predict_matrix(zac_features, 'clfs/softmax_grad.clf')
zac_knn_scores = clf_predict_matrix(zac_features, 'clfs/softmax_kn.clf')
zac_adaboost_scores = clf_predict_matrix(zac_features, 'clfs/softmax_ada.clf')

# manisha_rf_scores = clf_predict_matrix(manisha_features, 'clfs/autoenc_rf.clf')
# manisha_boost_scores = clf_predict_matrix(manisha_features, 'clfs/autoenc_grad.clf')
# manisha_knn_scores = clf_predict_matrix(manisha_features, 'clfs/autoenc_kn.clf')
# manisha_adaboost_scores = clf_predict_matrix(manisha_features, 'clfs/autoenc_ada.clf')

final_scores = zac_rf_scores + zac_knn_scores + zac_adaboost_scores + zac_scores + zac_scores + zac_scores

final_scores /= 6.0

np.savetxt('small_scores.txt', np.array(final_scores), newline='\n')
