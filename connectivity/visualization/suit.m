function varargout=run_suit(what,varargin)

%========================================================================================================================
%% (1) Directories
% BASE_DIR          = '/global/scratch/maedbhking/projects/cerebellum_language'; 
BASE_DIR            = '/Users/maedbhking/Documents/cerebellum_language_run'; 

DATA_DIR          = fullfile(BASE_DIR, 'data'); 
GLM_DIR           = fullfile(DATA_DIR, 'glm_firstlevel');  
SUIT_DIR          = fullfile(DATA_DIR, 'suit');
ANAT_DIR          = fullfile(DATA_DIR, 'anat'); 

subj_names = {'sub-01', 'sub-02', 'sub-03'}; % 'sub-01', 'sub-02','sub-03'

%% add software to search path
% addpath(genpath('/global/home/users/maedbhking/spm12'))
 addpath(genpath('/Users/maedbhking/Documents/MATLAB/spm12'))

%% functions
switch(what)

    case 'SUIT:run_all'                      % STEP 9.1-8.5
        %spm fmri
        space_label = "space-MNI152NLin2009cAsym_desc-preproc";  
        glm='glm2'; 
         
        for s=1:length(subj_names),
%             run_suit('SUIT:isolate_segment',s ,space_label)
%             run_suit('SUIT:correct_cereb_mask',s, space_label);
%             run_suit('SUIT:normalise_dartel',s, space_label);
%             run_suit('SUIT:reslice',s,'contrast', space_label, glm);clc
        end
        
        % map vol 2 surf for `folders`
        folders = dir(fullfile(GLM_DIR, glm)); 
        for i=3:length(folders),
            run_suit('SUIT:vol2surf', glm,folders(i).name)
        end

    case 'SUIT:isolate_segment'              % STEP 9.2:Segment cerebellum into grey and white matter
        sn=varargin{1};
        space_label=varargin{2}; 

        anat_name = sprintf('%s_%s_T1w.nii', subj_names{sn}, space_label); 
        SUIT_SUBJ_DIR = fullfile(SUIT_DIR, "anatomical", subj_names{sn});dircheck(SUIT_SUBJ_DIR);
        SOURCE=fullfile(ANAT_DIR, subj_names{sn}, anat_name); 
        DEST=fullfile(SUIT_SUBJ_DIR, anat_name);
        copyfile(SOURCE, DEST);
        cd(fullfile(SUIT_SUBJ_DIR));
        fprintf('starting the isolation and segmentation for %s', subj_names{sn})
        suit_isolate_seg({fullfile(SUIT_SUBJ_DIR, anat_name)},'keeptempfiles',1);

    case 'SUIT:correct_cereb_mask'           % STEP 9.4:
        sn=varargin{1}; 
        space_label=varargin{2}; 
            
        cortexGrey= fullfile(SUIT_DIR,"anatomical", subj_names{sn}, sprintf('c7%s_%s_T1w.nii',subj_names{sn}, space_label)); % cortex grey matter mask 
        cerebGrey = fullfile(SUIT_DIR,"anatomical", subj_names{sn}, sprintf('c1%s_%s_T1w.nii', subj_names{sn}, space_label)); % cerebellum grey matter mask
        bufferVox = fullfile(SUIT_DIR,"anatomical", subj_names{sn},'buffer_voxels.nii');

        % isolate overlapping voxels
        spm_imcalc({cortexGrey,cerebGrey},bufferVox,'(i1.*i2)')

        % mask buffer
        spm_imcalc({bufferVox},bufferVox,'i1>0')

        cerebGrey2 = fullfile(SUIT_DIR, "anatomical", subj_names{sn},'cereb_prob_corr_grey.nii');
        cortexGrey2= fullfile(SUIT_DIR, "anatomical", subj_names{sn},'cortical_mask_grey_corr.nii');

        % remove buffer from cerebellum
        spm_imcalc({cerebGrey,bufferVox},cerebGrey2,'i1-i2')

        % remove buffer from cortex
        spm_imcalc({cortexGrey,bufferVox},cortexGrey2,'i1-i2')  

    case 'SUIT:normalise_dartel'             % STEP 9.5: Normalise the cerebellum into the SUIT template.
        % Normalise an individual cerebellum into the SUIT atlas template
        % Dartel normalises the tissue segmentation maps produced by suit_isolate
        % to the SUIT template
        % !! Make sure the mask has been corrected !! (see 'SUIT:correct_cereb_mask')
        sn=varargin{1}; %subjNum
        space_label = varargin{2}; 
       
        cd(fullfile(SUIT_DIR, "anatomical", subj_names{sn}));
        job.subjND.gray      = {sprintf('c_%s_%s_T1w_seg1.nii', subj_names{sn}, space_label)};
        job.subjND.white     = {sprintf('c_%s_%s_T1w_seg2.nii', subj_names{sn}, space_label)};
        job.subjND.isolation= {'cereb_prob_corr_grey.nii'};
        suit_normalize_dartel(job);

    case 'SUIT:reslice'                      % STEP 9.8: Reslice the contrast images from first-level GLM
        % Reslices the functional data (betas, contrast images or ResMS)
        % from the first-level GLM using deformation from
        % 'suit_normalise_dartel'.
        % make sure that you reslice into 2mm^3 resolution
        sn=varargin{1}; % subjNum
        type=varargin{2}; % 'contrast'
        space_label=varargin{3};
        glm=varargin{4}; 
        
        mask = 'cereb_prob_corr_grey.nii';
        switch type
            case 'contrast'
                GLM_SUBJ_DIR = fullfile(GLM_DIR, glm, subj_names{sn});
                SUIT_SUBJ_DIR=fullfile(SUIT_DIR, 'anatomical', subj_names{sn});
                source=dir(fullfile(GLM_SUBJ_DIR,'*nii*')); % images to be resliced
                cd(GLM_SUBJ_DIR);
        end
        
        job.subj.affineTr = {fullfile(SUIT_SUBJ_DIR,sprintf('Affine_c_%s_%s_T1w_seg1.mat', subj_names{sn}, space_label))};
        job.subj.flowfield= {fullfile(SUIT_SUBJ_DIR,sprintf('u_a_c_%s_%s_T1w_seg1.nii', subj_names{sn}, space_label))};
        job.subj.resample = {source.name};
        job.subj.mask     = {fullfile(SUIT_SUBJ_DIR, mask)};
        job.vox           = [2 2 2];
        suit_reslice_dartel(job);
        % move resliced images to SUIT folder
        source=fullfile(GLM_SUBJ_DIR,'*wd*'); 
        SUIT_SUBJ_DIR_FUNC = fullfile(SUIT_DIR, 'functional', glm, subj_names{sn}); dircheck(SUIT_SUBJ_DIR_FUNC);
        movefile(source,SUIT_SUBJ_DIR_FUNC);
        fprintf('%s have been resliced into suit space for %s \n\n',type,subj_names{sn})

    case 'SUIT:vol2surf'                     % STEP 9.9: Make gifti files
        glm=varargin{1}; % 'glm1' etc
        suit_folder=varargin{2}; % 'group' or 'sub-01', 'sub-02' etc
        
        % map volume to surface and save out as gifti
        SUIT_SUBJ_DIR_FUNC = fullfile(SUIT_DIR, 'functional', glm, suit_folder);
        source=dir(fullfile(SUIT_SUBJ_DIR_FUNC,'*.nii*'));
        
        for m=1:length(source), 
            C=suit_map2surf(fullfile(SUIT_SUBJ_DIR_FUNC, source(m).name),'stats','nanmean');
        
            % make gifti structure and save out
            g = gifti(C);
            out_name = strrep(source(m).name,'.nii','');
            out_path = fullfile(SUIT_SUBJ_DIR_FUNC, strcat(out_name, '.gii'));
            save(g,out_path,'Base64Binary');
        end   
end

%% Local functions
function dircheck(dir)
if ~exist(dir,'dir');
    warning('%s doesn''t exist. Creating one now. You''re welcome! \n',dir);
    mkdir(dir);
end

