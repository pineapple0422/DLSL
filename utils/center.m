function [ prototype] = center( features, labels, class_labels)
dim = size(features,1);
Num = length(class_labels);
prototype = zeros(dim,Num);

for i = 1:Num
    index = labels==class_labels(i);
    current = features(:,index);
    prototype(:,i) = mean(current,2);
end

end