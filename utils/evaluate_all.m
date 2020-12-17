function [ ac_v, ac_a, ac_s, ac_va, ac_as, ac_vs, ac_vas ] = evaluate_all( features, Pu, Zu, labels, semantic,ClassLabels, D1, D2, gamma)
image_v = features;
image_a = learn_coefficients_noise(D1, image_v, gamma);
image_s = D2 * image_a;

class_s = semantic;
class_a = learn_coefficients_noise(D2, class_s, gamma);
class_v = D1*class_a;

score_v = score_eval(image_v, class_v);
ac_v = evaluate_ac(score_v, labels, ClassLabels);

score_a = score_eval(image_a, class_a);
ac_a = evaluate_ac(score_a, labels, ClassLabels);

score_s = score_eval(image_s, class_s);
ac_s = evaluate_ac(score_s, labels, ClassLabels);

score_va = score_v + score_a;
ac_va = evaluate_ac(score_va, labels, ClassLabels);

score_as = score_s + score_a;
ac_as = evaluate_ac(score_as, labels, ClassLabels);

score_is = score_v + score_s;
ac_vs = evaluate_ac(score_is, labels, ClassLabels);

score_vas = score_v + score_a + score_s;
ac_vas = evaluate_ac(score_vas, labels, ClassLabels);

end

function [value] = score_eval(image, proto)
sim = proto'*image;
a = sqrt(sum(proto.^2));
b = sqrt(sum(image.^2));
value = sim./(a'*b);
end

function [ac] = evaluate_ac(value, labels, ClassLabels)
[~,id] = max(value);
pre = ClassLabels(id);

avg_acc = 0;
for i = 1:length(ClassLabels)
    ind = labels==ClassLabels(i);
    temp = sum(pre(ind)==labels(ind))/sum(ind);
    avg_acc = avg_acc+temp;
end
ac = avg_acc/length(ClassLabels);
end