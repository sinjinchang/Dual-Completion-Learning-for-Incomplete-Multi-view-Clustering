datafile= '.\2v_VOC.mat'; % specify the file path and name
load(datafile);
save(filename, 'myMatrix'); % save the matrix to the .mat file
