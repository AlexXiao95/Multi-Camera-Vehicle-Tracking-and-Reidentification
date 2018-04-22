clear,close all
files = {'googlenet_v4.log.test',...
    'googlenet_v4_continue.log.test',...
    'googlenet_v4_soft.log.test',...
    'googlenet_v4_soft_combined.log.test',...
    'googlenet_v4_ext_label.log.test',...
	'googlenet_v4_ext_label_combined.log.test',...
    'googlenet_v4_ext_label_combined_continue.log.test',...
    
    };
settings = {'baseline', 'soft target',...
    'soft target with class label','extended label','extended label with class label',...
           };
colors = {'g','c','m','black','b'};
h = figure;
iter_offset=[0,265000,0,0,200000,0,160000];
iter_scale=[1,1,2,1,1,1,1];
file_model_idx = [1,1,2,3,4,5,5];
model_n = 5;
n_per_line = 8;
%organize data
iters = cell(model_n,1);
top5s = cell(model_n,1);
for i=1:length(files)
   [iter,top1,top5] = parse_log(files{i},8);
   iter = (iter+iter_offset(i))*iter_scale(i);%continue training
   iters{file_model_idx(i)} = [iters{file_model_idx(i)};iter];
   top5s{file_model_idx(i)} = [top5s{file_model_idx(i)};top5];
end

%plot
for i=1:model_n
   hold on;
   plot(iters{i},top5s{i},colors{i});
   hold off;
end
axis([0 1e6 0 1]);
xlabel('Iteration');
ylabel('Top-5 accuracy');
legend(settings,'Location','SouthEast');
saveas(gcf,'curve','png');
