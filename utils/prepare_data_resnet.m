function [ train_feature, test_feature , train_label, test_label, trainClassLabels, testClassLabels, semantic,name] = prepare_data_resnet(path,dataset)
dataset_path = [path '/' dataset];
load(sprintf('%s/att_splits.mat', dataset_path));
load(sprintf('%s/res50.mat', dataset_path));
train_feature = features(:,trainval_loc);
test_feature = features(:,test_unseen_loc);

train_label =  labels(trainval_loc);
test_label = labels(test_unseen_loc);
trainClassLabels = unique(train_label);
testClassLabels = unique(test_label);
semantic = att';
name = allclasses_names;

end

