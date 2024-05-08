import pickle


def get_all_trained_models():

    ret_dict = dict()

    with open("static\\trainedmodels\\decision_tree_classifier_model.pkl", 'rb') as file:
        decision_tree = pickle.load(file)
        ret_dict["decision_tree"] = decision_tree
    
    
    return ret_dict