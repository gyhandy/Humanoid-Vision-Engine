% load mandrill
% X = ind2rgb(X,map);

// root = 'data\iLab\preprocessed_images\train\ori_resize';
// save_root = '\data\iLab\feature_images\color\';

root = '';
save_root = '';

imgDataDir = dir(root)
for i = 1:length(imgDataDir)
    if(isequal(imgDataDir(i).name, '.')||isequal(imgDataDir(i).name, '..')||~imgDataDir(i).isdir)
        continue;
    end
    
    save_folder = strcat(save_root, '\', imgDataDir(i).name)
    if ~exist(save_folder,'dir')
        mkdir(save_folder)
    end
    
    path_list = dir(fullfile(root, imgDataDir(i).name, '*.jpg'));
    file_names = {path_list.name}';
    
    len = length(path_list);
    for j = 1:len
        img_object_path = strcat(root, imgDataDir(i).name, '\', file_names(j))
        X = imread(img_object_path{1,1});

    %     h = figure; p = get(h,'Position'); set(h,'Position',[0.2 1 2 1].*p); xlabel('Original')
    %     subplot(1,3,1), image(X); axis equal
        Xscrambled = imscramble(X,0.8,'cutoff');
    %     Xscrambled2 = imscramble(X,0.8,'range');
    %     subplot(1,3,2), image(Xscrambled);  axis equal; xlabel('Scrambled with ''cutoff''')
    %     subplot(1,3,3), image(Xscrambled2);  axis equal; xlabel('Scrambled with ''range''')

        save_dir = strcat(save_folder, '\', file_names(j))
        new_save_dir = strrep(save_dir{1,1},'.jpg','.jpg');
        imwrite(Xscrambled, new_save_dir);
    end
end