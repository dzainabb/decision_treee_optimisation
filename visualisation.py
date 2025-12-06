#visualisarion
# does not work on mac - needs graphviz installed - try in windows

dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=x.columns,  
                                class_names=clf.classes_,
                                filled=True, rounded=True,  
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('decision_tree_obesity.png')