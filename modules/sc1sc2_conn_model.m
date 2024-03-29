function [ varargout ] = sc1sc2_conn_model( what, varargin )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%==========================================================================
% setting directories
% baseDir         = '/Volumes/MotorControl/data/super_cerebellum_new';
% baseDir         = '/Users/ladan/Documents/Project-Cerebellum/Cerebellum_Data';
baseDir         = '/srv/diedrichsen/data/super_cerebellum';
baseDir         = '/Volumes/diedrichsen_data$/data/super_cerebellum/'; 
atlasDir        = '/srv/diedrichsen/data/Atlas_templates/fs_LR_32';
wbDir           = fullfile(baseDir,'sc1','surfaceWB');
behavDir        = 'data';
imagingDir      = 'imaging_data';
suitDir         = 'suit';
regDir          = 'RegionOfInterest';
connDir         = 'conn_models'; % setting a different directory than in sc1sc2_connectivity
suitToolDir     = '/Users/ladan/Documents/MATLAB/suit';
encodeDir       = 'encoding';
beta_roiDir     = 'beta_roi';
%==========================================================================
% setting some variables
subj_name = {'s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11',...
    's12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22','s23','s24',...
    's25','s26','s27','s28','s29','s30','s31'};
returnSubjs=[2,3,4,6,8,9,10,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,30,31];

%==========================================================================
% ROIs
corticalParcels    = {'yeo7', 'yeo17','tessels0042','tessels0162','tessels0362',...
    'tessels0642','tessels1002','tessels1442'};
cerebellarParcels = {'cerebellum_suit'};

hemI    = {'L', 'R'}; % left and right hemisphere
hemName = {'CortexLeft', 'CortexRight'};

%==========================================================================
% variables
runLst  = 1:16;    %% run numbers to use in GLM
% ntp     = 598;     %% number of time points after eliminating the dummy scans

switch what
    case 'ROI:define_suit_group'               % Defines group suit ROI
        cd(fullfile(baseDir,'sc1','suit','anatomicals'));
        spmj_resample('cerebellarGreySUIT_corrected.nii','cerebellarGreySUIT3mm.nii',2/3); % REssample to 3mm
        R{1}.type  = 'image';
        R{1}.file  = 'cerebellarGreySUIT3mm.nii';
        R{1}.name  = sprintf('cerebellum_suit');
        R{1}.threshold = 0.25;
        R=region_calcregions(R);
        save(fullfile(baseDir,'sc1','RegionOfInterest','data','group','regions_cerebellum_suit.mat'),'R');
    case 'ROI:define_suit'                     % Defines individual regions from the suit ROI
        sn = returnSubjs;
        saveasimg = 0; % Automatixally save the ROI as an image?
        vararginoptions(varargin, {'sn','saveasimg'});
        load(fullfile(baseDir,'sc1','RegionOfInterest','data','group','regions_cerebellum_suit.mat'));
        Rsuit = R{1}; 
        for s=sn
            suitDir=fullfile(baseDir,'sc1','suit','anatomicals',subj_name{s});
            fprintf([suitDir '\n']);
            [defs,mat]=spmdefs_get_dartel(fullfile(suitDir,'u_a_c_anatomical_seg1.nii'),fullfile(suitDir,'Affine_c_anatomical_seg1.mat'));
            % Deform the group region
            [R{1}.data(:,1),R{1}.data(:,2),R{1}.data(:,3)]=spmdefs_transform(defs,mat,Rsuit.data(:,1),Rsuit.data(:,2),Rsuit.data(:,3));
            save(fullfile(baseDir,'sc1','RegionOfInterest','data',subj_name{s},'regions_cerebellum_suit.mat'),'R');
            if (saveasimg)
                V=spm_vol(fullfile(baseDir,'sc1','imaging_data',subj_name{s},'rmeanepi.nii'));
                name = fullfile(baseDir,'sc1','RegionOfInterest','data',subj_name{s},'regions_cerebellum_suit.nii');
                region_saveasimg(R{1},V,'name',name);
            end
        end
    case 'ROI:define_labels'                   % Make group label files for the cortical ROIs
        atlas = varargin{1};    % Name of the atlas label file 
        name  = varargin{2};    % Name of the resultant group label file 
        sep_hem = varargin{3}; % Separate hemispheres? 
        offset = 0; 
        for h=1:2 
            in_name{h}=fullfile(atlasDir,sprintf('%s.%s.label.gii',atlas,hemI{h}));
            out_name{h}=fullfile(baseDir,'sc1',regDir,'data','group',sprintf('%s.%s.label.gii',name,hemI{h}));
            G{h}=gifti(in_name{h}); 
            [numVert,key]=pivottable(G{h}.cdata,[],G{h}.cdata,'length');
            highestROI= max(G{h}.cdata);
            if sum(key>0)<highestROI
                error('Labeling is not continous');
            end
            data = G{h}.cdata; 
            data(data>0) = data(data>0)+offset; 
            names = G{h}.labels.name;  
            G{h} = surf_makeLabelGifti(data,'labelNames',names,'labelRGBA',G{h}.labels.rgba,...
                'anatomicalStruct',surf_getGiftiAnatomicalStruct(G{h}));
            save(G{h},out_name{h});
            if sep_hem 
                offset = offset + highestROI;
            end 
        end
    case 'ROI:define_labels_all'
        sc1sc2_conn_model('ROI:define_labels','Icosahedron-42.32k','tessels0042',1);
        sc1sc2_conn_model('ROI:define_labels','Icosahedron-162.32k','tessels0162',1);
        sc1sc2_conn_model('ROI:define_labels','Icosahedron-362.32k','tessels0362',1);
        sc1sc2_conn_model('ROI:define_labels','Icosahedron-642.32k','tessels0642',1);
        sc1sc2_conn_model('ROI:define_labels','Icosahedron-1002.32k','tessels1002',1);
        sc1sc2_conn_model('ROI:define_labels','Icosahedron-1442.32k','tessels1442',1);
        sc1sc2_conn_model('ROI:define_labels','Yeo_JNeurophysiol11_7Networks.32k','yeo7',0);
        sc1sc2_conn_model('ROI:define_labels','Yeo_JNeurophysiol11_17Networks.32k','yeo17',0);

    case 'ROI:define_cortical'                 % Creates individual regions from cortical label files. 
        % defines the cortical ROIs using atlases' workbench gifti files for each subject
        atlas_res      = 32; 
        sn             = returnSubjs;
        parcelName     = 'tessels0162'; %% set it to 'tesselsWB', 'yeo_7', or 'yeo_17'
        
        vararginoptions(varargin, {'sn', 'glm', 'parcelName'});
        
        % setting glm directory
        %         glmDir = fullfile(baseDir, experiment, sprintf('GLM_firstlevel_%d', glm));
        surfDir = fullfile(baseDir, 'sc1', sprintf('fs_LR_%d', atlas_res)); %% the directory where the atlas is located
        
        for s = sn
            fprintf('Defining %s for %s \n', parcelName, subj_name{s})
            
            %             mask = fullfile(glmDir, subj_name{s}, 'mask.nii');
            mask = fullfile(baseDir, 'sc1', regDir,'data',subj_name{s}, 'cortical_mask_grey_corr.nii');
            subjWbDir = fullfile(wbDir,subj_name{s});
                
            for h = 1:2 % two hemispheres
                RR = []; %% initializing the R variable which will contain the data for the subject's ROI

                G = gifti(fullfile(baseDir,'sc1',regDir,'data','group',sprintf('%s.%s.label.gii',parcelName,hemI{h}))); 
                regs = unique(G.cdata)';  % exclude 0 - medial wall
                regs(regs==0)=[]; 
                for r=regs
                    RR{r}.type     = 'surf_nodes_wb';
                    RR{r}.location = find(G.cdata(:,1) == r);
                    RR{r}.white    = fullfile(subjWbDir,sprintf('%s.%s.white.%dk.surf.gii',subj_name{s},hemI{h}, atlas_res));
                    RR{r}.pial     = fullfile(subjWbDir,sprintf('%s.%s.pial.%dk.surf.gii',subj_name{s},hemI{h}, atlas_res));
                    RR{r}.linedef  = [5, 0, 1];
                    RR{r}.image    = mask;
                    RR{r}.name     = sprintf('%s.parcel-%0.2d', hemI{h}, r);
                end
                Rreg{h} = region_calcregions(RR);
            end % h (hemi)
            
            % Combine the regions across hemispheres 
            R = []; 
            for i=1:length(Rreg{2})
                if (length(Rreg{1})<i || isempty(Rreg{1}{i}))
                    R{i}=Rreg{2}{i}; 
                elseif (length(Rreg{1})<i || isempty(Rreg{2}{i})) 
                    R{i}=Rreg{1}{i};               
                else
                    R{i}=Rreg{1}{i};
                    R{i}.type='surf_nodes_bilateral';
                    R{i}.name = sprintf('parcel-%0.2d', r);
                    R{i}.data = [Rreg{1}{i}.data;Rreg{2}{i}.data];
                    R{i}.linvoxidxs = [Rreg{1}{i}.linvoxidxs;Rreg{2}{i}.linvoxidxs];
                end
            end 
            save(fullfile(baseDir, 'sc1', regDir, 'data', subj_name{s}, sprintf('regions_%s.mat',parcelName)),'R', '-v7.3');
            fprintf('\n');
        end % sn (subject)
    case 'ROI:define_cortical_all' 
        for i=1:length(corticalParcels)
            % 44,'parcelName',corticalParcels{i});
        end
    case 'ROI:beta_unn'
        % Calculate BetaUW for regions
        % univariately normalizes beta values and saves both the normalized
        % and non-normalized beta values in a new structure.
        % Changed this code to directly work on the beta images
        % for sc1, there are 88 beta values per subject.
        sn             = returnSubjs;
        experiment_num = 1;
        parcelType     = 'tessels0162';  %% other options are 'cerebellum_suit', 'yeo_7WB', and 'yeo_17WB'
        glm            = 7;
        ignore_nan     = 0; % Set Nans to zero for sampling? 
        interp         = 0; % Interpolation for sampling 
        
        vararginoptions(varargin, {'sn', 'experiment_num', 'glm', 'parcelType', 'ignore_nan','interp'});
        
        experiment = sprintf('sc%d', experiment_num);
        
        % setting directories
        
        for s = sn
            glmDir  = fullfile(baseDir, experiment, sprintf('GLM_firstlevel_%d', glm), subj_name{s});
            betaDir = fullfile(baseDir, experiment, beta_roiDir, sprintf('glm%d', glm), subj_name{s});
            fprintf('%s univariate normalized betas for %s \n', parcelType, subj_name{s});
            B = [];
            
            % load in SPM_info to get the info for the tasks and
            % condditions
            T           = load(fullfile(glmDir, 'SPM_info.mat'));
            
            % load in spm file
            % Checks if SPM_light exists. Loads it if its exists, otherwise
            % the bigger SPM.mat file will be loaded
            if exist(fullfile(glmDir, 'SPM_light.mat'), 'file')
                load(fullfile(glmDir, 'SPM_light.mat'));
            else
                load(fullfile(glmDir, 'SPM.mat'));
            end
            
            % load in ROI file
            roi_name=sprintf('regions_%s.mat',parcelType);
            load(fullfile(baseDir, 'sc1', regDir, 'data', subj_name{s}, roi_name));
            
            % Get the raw data files
            V=SPM.Vbeta;
            V(end+1)=SPM.VResMS;
            cd(glmDir);
            tic;
            Y = region_getdata(V,R,'interp',interp,'ignore_nan',ignore_nan);  % Data is N x P
            B=[];
            for r = 1:numel(R) % R is the output 'regions' structure from 'ROI_define'
                % Get betas (univariately prewhitened)
                resMS = Y{r}(end,:);
                beta  = Y{r}(1:end-1,:); % remove resMS

                B_tmp.betasNW{1, 1} = beta; 
                B_tmp.betasUW{1, 1} = bsxfun(@rdivide,beta,sqrt(resMS));
                if (ignore_nan) 
                    B_tmp.betasUW{1}(:,resMS==0)=0;
                end
                B_tmp.mbetasNW  = nanmean(B_tmp.betasNW{1},2)';
                B_tmp.mbetasUW  = nanmean(B_tmp.betasUW{1},2)';
                B_tmp.region_num    = r;
                B_tmp.region_name   = {R{r}.name};
                B_tmp.SN            = s;
                
                B = addstruct(B, B_tmp);
                roi_name = parcelType;
            end
            
            % save the betas
            dircheck(betaDir);
            save(fullfile(betaDir, sprintf('beta_regions_%s.mat', roi_name)), 'B', '-v7.3');
            fprintf('\n');
        end % sn
    case 'ROI:add_to_beta'               % creates a new structure using beta_region files.
        % The structure will be uesd in connectivity project only
        % Example: sc1sc2_conn_model('ROI:MDTB:add_to_beta', 'experiment_num', 1, 'parcelType', 'yeo_7WB', 'glm', 7)
        
        sn             = returnSubjs;
        experiment_num = 1;
        parcelType     = 'tessels0162';  %% other options are 'cerebellum_suit', 'yeo_7WB', and 'yeo_17WB'
        glm            = 7;
        %%% run 'ROI:mdtb:empty_parcel' to get oparcel.
        
        vararginoptions(varargin, {'sn', 'experiment_num', 'glm', 'parcelType', 'oparcel', 'discardp', 'which'});
        
        experiment = sprintf('sc%d', experiment_num);
        
        % setting directories
        glmDir  = fullfile(baseDir, experiment, sprintf('GLM_firstlevel_%d', glm));
        betaDir = fullfile(baseDir, experiment, beta_roiDir, sprintf('glm%d', glm));
        dircheck(betaDir); % beta values for each parcelType is saved here
        
        % For each subject, load SPM_info and B file and add the fields
        for s = sn
            Y = [];
            % load SPM_info
            fprintf('%s modifying B for %s %s \n', parcelType, subj_name{s});
            
            % load in SPM_info to get the info for the tasks and
            % condditions
            T  = load(fullfile(glmDir, subj_name{s}, 'SPM_info.mat'));
            
            % calculate the inner product of design matrix for proper weighting
            try
                load(fullfile(glmDir, subj_name{s}, 'SPM_light.mat'));
            catch
                load(fullfile(glmDir, subj_name{s}, 'SPM.mat'));
            end
            
            X= SPM.xX.X; % Use the unweigthed design matrix
            for i = 1:length(SPM.Sess)
                XB = X(SPM.Sess(i).row,SPM.Sess(i).col);
                XB = bsxfun(@minus,XB,mean(XB)); % Subtract the mean
                XX(:,:,i) = XB'*XB;
            end;
            
            % load Beta file
            load(fullfile(betaDir, subj_name{s}, sprintf('beta_regions_%s', parcelType)));
            
            % discarding the intercepts
           if length(B.betasUW) == 1 % Single region, preserve each voxel 
                myfield = 'betasUW';
                Y.data = B.(myfield){1}(1:end - 16, :); % discarding the intercepts
            else % Many regions - take the mean of each region 
                myfield = 'mbetasUW';
                tmp = B.(myfield)';
                Y.data = tmp(1:end - 16, :); % discarding the intercepts
            end
            
            % adding the extra fields from SPM_info
            Y = addstruct(Y, T);
            Y.XX = XX;
            
            % save the betas
            % dircheck(fullfile(betaDir, subj_name{s}));
            save(fullfile(betaDir, subj_name{s}, sprintf('Y_glm%d_%s.mat', glm, parcelType)), '-struct', 'Y', '-v7.3');
            fprintf('\n');
        end % s (sn)
    case 'ROI:beta_all' % extracts betas and add_to_beta
        % Example usage: sc1sc2_conn_model('ROI:MDTB:beta_all')
        sn = returnSubjs; 
        experiment_num=2;
        glm = 7; 
        % sc1sc2_conn_model('ROI:beta_unn', 'sn', sn, 'experiment_num', 1, 'glm', 7, 'parcelType', 'cerebellum_suit','interp',1,'ignore_nan',1);
        % sc1sc2_conn_model('ROI:beta_unn', 'sn', sn, 'experiment_num', 2, 'glm', 7, 'parcelType', 'cerebellum_suit','interp',1,'ignore_nan',1);
        sc1sc2_conn_model('ROI:beta_unn', 'sn', sn, 'experiment_num', experiment_num, 'glm', glm, 'parcelType', 'tessels0042');
        sc1sc2_conn_model('ROI:beta_unn', 'sn', sn, 'experiment_num', experiment_num, 'glm', glm, 'parcelType', 'tessels0162');
        sc1sc2_conn_model('ROI:beta_unn', 'sn', sn, 'experiment_num', experiment_num, 'glm', glm, 'parcelType', 'tessels0362');
        sc1sc2_conn_model('ROI:beta_unn', 'sn', sn, 'experiment_num', experiment_num, 'glm', glm, 'parcelType', 'tessels0642');
        sc1sc2_conn_model('ROI:beta_unn', 'sn', sn, 'experiment_num', experiment_num, 'glm', glm, 'parcelType', 'tessels1002');
        sc1sc2_conn_model('ROI:beta_unn', 'sn', sn, 'experiment_num', experiment_num, 'glm', glm, 'parcelType', 'yeo7');
        sc1sc2_conn_model('ROI:beta_unn', 'sn', sn, 'experiment_num', experiment_num, 'glm', glm, 'parcelType', 'yeo17');
        % add betas
        sc1sc2_conn_model('ROI:add_to_beta', 'sn', sn, 'experiment_num', experiment_num, 'glm', glm, 'parcelType', 'tessels0042');
        sc1sc2_conn_model('ROI:add_to_beta', 'sn', sn, 'experiment_num', experiment_num, 'glm', glm, 'parcelType', 'tessels0162');
        sc1sc2_conn_model('ROI:add_to_beta', 'sn', sn, 'experiment_num', experiment_num, 'glm', glm, 'parcelType', 'tessels0362');
        sc1sc2_conn_model('ROI:add_to_beta', 'sn', sn, 'experiment_num', experiment_num, 'glm', glm, 'parcelType', 'tessels0642');
        sc1sc2_conn_model('ROI:add_to_beta', 'sn', sn, 'experiment_num', experiment_num, 'glm', glm, 'parcelType', 'tessels1002');
        sc1sc2_conn_model('ROI:add_to_beta', 'sn', sn, 'experiment_num', experiment_num, 'glm', glm, 'parcelType', 'yeo7');
        sc1sc2_conn_model('ROI:add_to_beta', 'sn', sn, 'experiment_num', experiment_num, 'glm', glm, 'parcelType', 'yeo17');
        % sc1sc2_conn_model('ROI:add_to_beta', 'sn', sn, 'experiment_num', 1, 'glm', 7, 'parcelType', 'cerebellum_suit');
        % sc1sc2_conn_model('ROI:add_to_beta', 'sn', sn, 'experiment_num', 2, 'glm', 7, 'parcelType', 'cerebellum_suit');
  
        
    case 'PREP:MDTB:cortex:surface'           % creates Y_info file for the cortical surfaces
        % gets the betas on the surface and univariately prewhiten them. It
        % saves the beta values for the cortex in a new structure that also
        % has the task info (using SPM_info.mat to get the task info).
        % Example: sc1sc2_conn_model('PREP:MDTB:cortex:surface', 'experiment_num', 1, 'glm', 8)
        sn             = returnSubjs;
        experiment_num = 1;
        glm            = 7;
        atlas_res      = 32;
        which          = 'cond';
        
        vararginoptions(varargin, {'sn', 'experiment_num', 'glm', 'atlas_res', 'which'});
        
        experiment = sprintf('sc%d', experiment_num);
        
        % setting glm and surfaceWB directory
        glmDir      = fullfile(baseDir, experiment, sprintf('GLM_firstlevel_%d', glm));
        glmSurfDir  = fullfile(wbDir, sprintf('glm%d', glm)); dircheck(glmSurfDir);
        encodingDir = fullfile(baseDir, experiment, encodeDir, sprintf('glm%d', glm));
        
        numRuns = length(runLst);
        
        for s = sn
            Y = [];
            subjSurfDir = fullfile(wbDir, 'data', subj_name{s});
            dircheck(fullfile(glmSurfDir, subj_name{s})); %% directory to save the contrast maps
            
            glmSubjDir    = fullfile(glmDir, subj_name{s});
            encodeSubjDir = fullfile(encodingDir, subj_name{s});
            
            T = load(fullfile(glmDir, subj_name{s}, 'SPM_info.mat'));
            for h = 1:2 % two hemispheres
                white   = fullfile(subjSurfDir,sprintf('%s.%s.white.%dk.surf.gii',subj_name{s},hemI{h}, atlas_res));
                pial    = fullfile(subjSurfDir,sprintf('%s.%s.pial.%dk.surf.gii',subj_name{s},hemI{h}, atlas_res));
                C1      = gifti(white);
                C2      = gifti(pial);
                
                % map ResMS to the surface
                ResMsImage{1}    = fullfile(glmDir, subj_name{s}, 'ResMS.nii');
                ResMs_colName{1} = 'ResMS.nii';
                ResMs_outfile    = fullfile(glmSurfDir, subj_name{s}, sprintf('%s.%s.ResMS.func.gii', subj_name{s}, hemI{h}));
                
                G_ResMs = surf_vol2surf(C1.vertices,C2.vertices,ResMsImage,'column_names', ResMs_colName, ...
                    'anatomicalStruct',hemName{h});
                save(G_ResMs, ResMs_outfile);
                
                % start mapping betas to surface
                filenames = dir(fullfile(glmSubjDir,'beta*'));
                outfile   = fullfile(glmSurfDir, subj_name{s}, sprintf('%s.%s.glm%d_beta_cortex_%s.func.gii', ...
                    subj_name{s}, hemI{h},glm));
                
                for t = 1:length(filenames)
                    fileList{t}    = {filenames(t).name};
                    column_name{t} = {filenames(t).name};
                end % t
                for f=1:length(fileList)
                    images(f) = spm_vol(fullfile(glmSubjDir,fileList{f}));
                end % f
                G = surf_vol2surf(C1.vertices,C2.vertices,images,'column_names', column_name, ...
                    'anatomicalStruct',hemName{h});
                save(G, outfile);
                
                % write out new structure ('Y_info')
                Y.data = bsxfun(@rdivide, G.cdata', sqrt(G_ResMs.cdata')); % UW betas
                Y.data(end-numRuns+1:end, :)= []; % deleting the intercepts;
                
                Y            = addstruct(Y, T);
                Y.identity   = indicatorMatrix('identity',T.(which));
                %                 Y.nonZeroInd=B.index';
                
                outName = fullfile(encodeSubjDir,sprintf('Y_info_glm%d_cortex_%s.mat',glm,hemI{h}));
                save(outName,'Y','-v7.3');
                fprintf('cortical vertices: (Y data) computed for %s \n',subj_name{s});
                clear B R Y
            end % hemi
        end % sn
    case 'PREP:MDTB:Y_allsubjs'               % Make a structure of activity profiles of all subject
        % B, mean betas, and zero-meaned betas are stored
        % Example: sc1sc2_conn_model('PREP:MDTB:Y_allsubjs' , 'experiment_num', 1, 'glm', 7)
        
        sn               = returnSubjs; % Take only good subjects
        glm              = 7;           % glmNum
        experiment_num   = 1;           % either 1 or 2
        type             ='grey_nan';   % 'cortex_L', 'cortex_R', 'tesselsWB162', ...
        session_specific = 1;
        
        vararginoptions(varargin, {'sn', 'glm', 'experiment_num', 'type', 'session_specific'});
        
        experiment = sprintf('sc%d', experiment_num);
        
        % setting directories
        glmDir      = fullfile(baseDir, experiment, sprintf('GLM_firstlevel_%d', glm));
        encodingDir = fullfile(baseDir, experiment, encodeDir, sprintf('glm%d', glm));
        
        numSubj=length(sn);
        
        for si = 1:numSubj
            fprintf('subject %s\n',subj_name{sn(si)});
            
            % load condition data
            glmDirSubj = fullfile(glmDir, subj_name{sn(si)});
            T          = load(fullfile(glmDirSubj,'SPM_info.mat'));
            
            % load data
            filename = fullfile(encodingDir,subj_name{sn(si)},sprintf('Y_info_glm%d_%s.mat', glm, type));
            S        = load(filename);
            Y        = S.Y;
            
            % Decompose betas into mean and residuals
            if (session_specific == 1)
                Z =indicatorMatrix('hierarchicalI',[T.sess T.cond]);
            else
                Z =indicatorMatrix('identity',T.cond);
            end
            
            % remove mean
            B(:, :, si)   = Y.data;           %-beta
            Bm (:, :, si) = Z*pinv(Z)*Y.data; %-mean beta
            Bzc(:, :, si) = Y.data - Bm(:, :, si);        %-zero-centered beta
        end % si (numSubj)
        
        if (session_specific == 1)
            save(fullfile(encodingDir, sprintf('Y_%s_all_se.mat',type)),'B', 'Bm', 'Bzc', '-v7.3');
        else
            save(fullfile(encodingDir,sprintf('Y_%s_all.mat',type)), 'B', 'Bm', 'Bzc', '-v7.3');
        end
    case 'CONN:MDTB:CortCov'                  % Calculates and plots variance inflation factor
        % borrowed from sc1sc2_connectivity('cortical_covariances' )
        % Example: sc1sc2_conn_model('CONN:MDTB:CortCov', 'sn', 6, 'glm', 7, 'experiment_num', 1, 'xres', 162)
        sn             = returnSubjs;
        experiment_num = 1;
        glm            = 7;
        xres           = 162;
        
        atlas_res = 32; % other option is 64?
        
        vararginoptions(varargin,{'sn', 'experiment_num','glm','xres', 'atlas_res'});
        
        experiment = sprintf('sc%d', experiment_num);
        
        % setting directories
        betaDir = fullfile(baseDir, experiment, sprintf('Beta_GLM_%d', glm));
        surfDir = fullfile(baseDir, experiment, sprintf('fs_LR_%d', atlas_res)); %% the directory where the atlas is located + giftti label files
        %         glmDir  = fullfile(baseDir, experiment, sprintf('GLM_firstlevel_%d', glm));
        connDir = fullfile(baseDir, experiment, 'connModels');
        
        %         % Figure out the right ones to use ????
        %         X  = load(fullfile(baseDir, experiment, regDir,'data', sprintf('%d_reorder.mat', xres)));
        %         Xx = getrow(X,X.newIndx);
        %         Xx = getrow(Xx,Xx.good == 1);
        
        % Calculate correlations
        is = 1;
        for s = sn
            fprintf('Cortical covarianes for subject: %s\n', subj_name{s});
            % load Individual data
            filename = fullfile(betaDir, subj_name{s}, sprintf('beta_regions_tesselsWB%d', xres));
            load(filename);
            
            %             B_good = B(:, Xx.regIndx2);   % Remove the bad regions
            %             B2 = bsxfun(@minus, B_good, nanmean(B_good));
            %
            % get the tessels mean beta: mbetasUW field
            Betas = (B.mbetasUW)';
            
            % remove the intercepts
            Betas = Betas(1:end - length(runLst), :);
            
            % mean centre the Betas
            % this can be done either run-wise or as a whole
            BetasC = bsxfun(@minus, Betas, nanmean(Betas));
            
            % Covariance and correlation
            COV(:,:,is) = cov(BetasC);
            COR(:,:,is) = corrcov(COV(:,:,is));
            
            % Variance inflation factor for each
            AP.SN(is,1)   = s;
            AP.VAR(is,:)  = diag(COV(:,:,is));       % Variance of the task-related betas: How much activity could we induce?
            AP.VIF(is,:)  = diag(inv(COR(:,:,is)))'; % Variance inflation factor: How much does the region suffer from co-linearity?
            AP.VARB(is,:) = diag(inv(COV(:,:,is)));  % Combination of the last two: How well would we estimate connectivity weights from this region?
            
            is = is + 1;
        end % s (sn)
        
        % Plot group mean in flatmap
        data{1} = sqrt(mean(AP.VAR,1));
        data{2} = sqrt(mean(AP.VIF,1));
        data{3} = sqrt(mean(AP.VARB,1));
        
        plottitle{1} = 'Variance';
        plottitle{2} = 'Variance Inflation Factor';
        plottitle{3} = 'Combined';
        
        % using workbench
        % lots of this part is like the ROI:define
        %%% saves Variance, VIF, and combined as a single gifti file which
        %%% is saved under directory connModels
        idx = 1;
        for h = 1:2
            
            G = gifti(fullfile(surfDir,sprintf('Icosahedron-%d.%dk.%s.label.gii', xres, atlas_res, hemI{h})));
            
            nReg = numel(unique(G.cdata))-1; % exclude 0 - medial wall
            
            dv = zeros(length(G.cdata), 3);
            for r = 1:nReg
                dv(G.cdata(:,1) == r, 1) = data{1}(idx);
                dv(G.cdata(:,1) == r, 2) = data{2}(idx);
                dv(G.cdata(:,1) == r, 3) = data{3}(idx);
                idx = idx + 1;
            end % r(tessels)
            
            G       = surf_makeFuncGifti(dv,'anatomicalStruct', hemName{h}, 'columnNames', plottitle);
            outfile = (fullfile(connDir, sprintf('CortCov-%d.glm%d.%dk.%s.func.gii', xres, glm, atlas_res, hemI{h})));
            save(G, outfile);
        end % h (hemispheres)
    case 'CONN:MDTB:CortCov_all'              % Calculates Variance, VIF for all the tesselations
        % Example: sc1sc2_conn_model('CONN:MDTB:CortCov_all', 'sn', [2,3,4,6,8,9,10,12,14,15,17], 'xresArr', [162, 362, 642])
        
        sn             = returnSubjs;
        experiment_num = 1;
        glm            = 7;
        xresArr        = [162, 362, 642];
        
        vararginoptions(varargin, {'sn', 'experiment_num', 'glm', 'xresArr'})
        
        for nn = xresArr
            fprintf('Cortical covariances for %d tesselation\n', nn)
            sc1sc2_conn_model('cortical_covariances', 'sn', sn, 'glm', glm, 'experiment_num', experiment_num, 'xres', nn);
        end % nn
        
    case 'CONN:MDTB:model_mbeta'              % Conn. modelling using average beta per session
        % uses the mat file saved in the case 'get_meanBeta_cerebellum'
        % Example: sc1sc2_conn_model('CONN:MDTB:model_mbeta', 'method', 'linRegress')
        
        sn         = returnSubjs;
        glm        = 7;
        method     = 'plsr';         % linRegress, ridgeFixed, plsr, pcr
        yname      = 'grey_nan';     % cerebellar grey matter
        xname      = 'tesselsWB162'; % cortical tessels used for modelling: tesselsWB362, tesselsWB642
        trainMode  = 'crossed';      % training can be done in a crossed or uncrossed fashion
        overwrite  = 0;              % Overwrite old results or add to existing structure?
        trainExper = 1;              % Task set to train
        inclInstr  = 1;              % For future use: include Instruction
        params     = 10;             % params are to be specified according to the method. The default is for pls regression
        scale      = 0;              % scale=1: zscore X and Y (particularly for pls regression)
        %         numReg   = NaN;%????
        
        vararginoptions(varargin,{'sn', 'method', 'yname', 'xname', 'trainMode', ...
            'overwrite', 'trainExper', 'inclInstr', 'params', 'scale', 'glm'});
        
        experStr = {'sc1', 'sc2'}; % experiment strings
        
        % Setting directories
        name     = sprintf('mb4_%s_%s', xname, method);
        outDir   = fullfile(baseDir, experStr{trainExper}, connDir, sprintf('glm%d', glm), name);
        dircheck(outDir);
        
        % Get data
        [Y, ~] = sc1sc2_conn_model('PREP:MDTB:get_wcon', 'type', yname, 'inclInstr',inclInstr, 'experStr', experStr(trainExper)); % get the data for the cerebellum
        [X, S] = sc1sc2_conn_model('PREP:MDTB:get_wcon', 'type', xname, 'inclInstr',inclInstr, 'experStr', experStr(trainExper)); % get the data for cortical tessels
        
        % Find the data that we want to use for fitting the connectivity
        % model
        SI1 = find(ismember(S.StudyNum,trainExper) & S.sess==1);
        SI2 = find(ismember(S.StudyNum,trainExper) & S.sess==2);
        
        trainXindx = [SI1; SI2];
        switch (trainMode)
            case 'crossed'
                trainYindx = [SI2; SI1];
            case 'uncrossed'
                trainYindx = [SI1; SI2];
        end
        
        % Estimate the model and store the information attached to it
        RR    = [];
        for si = 1:length(sn)
            % Find relevant file
            fprintf('subject: %d',sn(si));
            outName = fullfile(outDir,sprintf('%s_%s.mat',name,subj_name{sn(si)}));
            
            % add the new model to the previous one or over-write it?
            if (exist(outName, 'file') && overwrite==0)
                R = load(outName);
            else
                R = [];
            end
            
            % Get data
            ss = find(sn(si) == returnSubjs);
            xx = X{ss}(trainXindx,:);
            yy = Y{ss}(trainYindx,:);
            
            [rp, cp] = size(params);
            
            % Run for all the params
            tmpR = []; % will contain the model parameters for each subject
            fprintf(' parameter: ');
            if ~isempty(params)
                for ip = 1:cp
                    fprintf('%d %d; ',params(:, ip));
                    
                    [W, sR2, sR, sR2_vox, sR_vox, varout] = conn_model_fit(yy,xx,method, 'params', params(:, ip), 'scale', scale);
                    
                    tmpR.SN           = sn(si);
                    tmpR.W{1, 1}      = W;
                    tmpR.params{1, 1} = params(:, ip);
                    tmpR.method       = {method};
                    tmpR.incInstr     = inclInstr;
                    tmpR.trainMode    = {trainMode};
                    tmpR.xname        = {xname};
                    %             R.numReg(i,1) = i;
                    tmpR.R2           = sR2;
                    tmpR.R            = sR;
                    tmpR.R2vox        = sR2_vox;
                    tmpR.Rvox         = sR_vox;
                    tmpR.varout       = varout;
                    
                    R    = addstruct(R, tmpR);
                end % p (params)
            elseif (isempty(params))
                fprintf('no params! ');
                
                [W, sR2, sR, sR2_vox, sR_vox, varout] = conn_model_fit(yy,xx,method, 'params', params, 'scale', scale);
                
                tmpR.SN           = sn(si);
                tmpR.W{1, 1}      = W;
                tmpR.params{1, 1} = params;
                tmpR.method       = {method};
                tmpR.incInstr     = inclInstr;
                tmpR.trainMode    = {trainMode};
                tmpR.xname        = {xname};
                %             R.numReg(i,1) = i;
                tmpR.R2           = sR2;
                tmpR.R            = sR;
                tmpR.R2vox        = sR2_vox;
                tmpR.Rvox         = sR_vox;
                tmpR.varout       = varout;
                
                R    = addstruct(R, tmpR);
                
            end
            
            RR    = addstruct(RR,R);
            save(outName,'-struct','R', '-v7.3');
            fprintf('\n');
        end % si (subjects)
        varargout{1} = {RR};
    case 'CONN:MDTB:run_model_mbeta'          % Runs different models
        sc1sc2_conn_model('CONN:MDTB:model_mbeta', 'method', 'linRegress', 'params', []);
        sc1sc2_conn_model('CONN:MDTB:model_mbeta', 'method', 'ridgeFixed', 'sn', subs, 'params', [0, 20], 'overwrite', 1);
        sc1sc2_conn_model('CONN:MDTB:model_mbeta', 'method', 'pcr', 'sn', subs, 'params', [10, 12], 'overwrite', 1, 'scale', 1);
        sc1sc2_conn_model('CONN:MDTB:model_mbeta', 'method', 'plsr_1', 'sn', subs, 'params', [10, 12], 'overwrite', 1, 'scale', 1);
        
    case 'CONN:MDTB:sample_DesignMatrix'      % saves a sample design matrix for weighting estimates later during evaluation
        % later in the python code this design matrix will be needed. It
        % would be better to save this design matrix so that I don't need
        % to load the whole SPM.mat structure in python. So far, today is
        % June 22 2020, I haven't figured out a way to import SPM.mat
        % structure into python!
        % Example: sc1sc2_conn_model('CONN:MDTB:sample_DesignMatrix', 'sn', 2, 'experNums', [1], 'glm', 7)
        
        sn        = returnSubjs;
        experNums = [1, 2];
        glm       = 7;
        
        vararginoptions(varargin, {'sn', 'experNums', 'glm'})
        
        experStr = {'sc1', 'sc2'};
        
        for e = experNums
            fprintf('Doing experiment %d\n', e);
            glmDir   = fullfile(baseDir,experStr{e},sprintf('GLM_firstlevel_%d',glm));
            for s = sn
                fprintf('Doing subject %s\n', subj_name{s});
                % load in spm file
                % Checks if SPM_light exists. Loads it if its exists, otherwise
                % the bigger SPM.mat file will be loaded
                if exist(fullfile(glmDir, subj_name{s}, 'SPM_light.mat'), 'file')
                    load(fullfile(glmDir, subj_name{s}, 'SPM_light.mat'));
                else
                    load(fullfile(glmDir, subj_name{s}, 'SPM.mat'));
                end
                spm.Sess = SPM.Sess;      % Session Structure
                spm.X    = SPM.xX.X;      % Design matrix
                spm.wX   = SPM.xX.xKXs.X; % Filtered design matrix
                
                save(fullfile(glmDir, subj_name{s},'SampleDesignMatrix.mat'), '-struct', 'spm', '-v7');
                
            end % s(sn)
        end
        
    case 'CONN:MDTB:evaluate'                 % Evaluates predictive performance of connectivity model
        % borrowed from sc1sc2_connectivity('evaluate')
        % load the model into workspace first! Make sure that the loaded
        % variable has all the models (different sets of parameters) you
        % want to evaluate!
        % Example: sc1sc2_conn_model('CONN:MDTB:evaluate', M);
        
        M         = varargin{1};     %% M: is the model structure with the connectivity weights in it (M.W)
        subset    = [];              %% 'subset': what data should the evaluation be based upon?
        splitby   = [];              %% 'splitby': by which variable should the evaluation be split?
        yname     = 'grey_nan';
        xname     = 'tesselsWB162';
        inclInstr = 1;
        meanSubt  = 1;               %% Mean pattern subtraction before prediction?
        
        vararginoptions(varargin(2:end),{'subset','splitby','meanSubt','xname','yname'});
        
        %         experStr = {'sc1', 'sc2'};
        
        % Get data
        [Y, ~] = sc1sc2_conn_model('PREP:MDTB:get_wcon', 'type', yname, 'inclInstr',inclInstr); % get the data for the cerebellum
        [X, S] = sc1sc2_conn_model('PREP:MDTB:get_wcon', 'type', xname, 'inclInstr',inclInstr); % get the data for cortical tessels
        
        %         % Get task/Condition info
        %         T = dload(fullfile(baseDir,'sc1_sc2_taskConds2.txt'));
        
        if (isempty(subset))
            subset = S.StudyNum == 2;
        end
        %         S.subset = [subset; subset]; % ??????? This might be wrong. Need to check it
        S.subset = subset;
        
        if (isempty(splitby))
            splitby = ones(length(S.StudyNum),1);
        end
        %         S.splitby = [splitby; splitby];  % Double for the two conditions???????
        S.splitby = splitby;  % Double for the two conditions???????
        
        sS     = getrow(S, S.subset);
        splits = unique(sS.splitby);
        
        % Loop over models and evaluate
        numModels = size(M.SN, 1);
        
        RR = [];
        for m = 1:numModels
            s = find(returnSubjs == M.SN(m)); % this makes sure that only the good subjects are used!
            
            goodindx = sum(abs(Y{s}(:,:)))>0 & ~isnan(sum(Y{s}(:,:))) & ~isnan(sum(M.W{m})); %?????????
            
            for sp = 1:length(splits)
                if meanSubt == 1
                    for sess = 1:2
                        indx = find(S.subset & S.sess == sess & S.splitby == splits(sp));
                        X{s}(indx,:) = bsxfun(@minus,X{s}(indx,:),mean(X{s}(indx,:)));
                        Y{s}(indx,:) = bsxfun(@minus,Y{s}(indx,:),mean(Y{s}(indx,:)));
                    end
                end
                
                % Make one index that arranges the data [sess1;sess2]
                testAindx=[find(S.subset & S.sess == 1 & S.splitby == splits(sp));...
                    find(S.subset & S.sess == 2 & S.splitby == splits(sp))];
                
                % Make one index that arranges the data [sess2;sess1]
                testBindx=[find(S.subset & S.sess == 2 & S.splitby == splits(sp));...
                    find(S.subset & S.sess == 1 & S.splitby == splits(sp))];
                
                % If the data is sorted [sess2;sess1], then [sess1;sess2]
                % is the crossvalidated estimate.
                predY   = X{s}(testAindx,:)*M.W{m};                    % Predicted Y using crossvalidation
                predYnc = X{s}(testBindx,:)*M.W{m};                    % Predicted Y not crossvalidated
                
                % Caluculate the respective sums-of-squares
                SSP   = sum(sum(predY(:,goodindx).^2));                                % Sum of square of predictions
                SSY   = sum(sum(Y{s}(testBindx,goodindx).^2));                         % Sum of squares of data
                SSCp  = sum(sum(predY(:,goodindx).*predYnc(:,goodindx)));              % Covariance of Predictions
                SSCy  = sum(sum(Y{s}(testAindx,goodindx).*Y{s}(testBindx,goodindx)));  % Covariance of Y's
                SSCn  = sum(sum(predYnc(:,goodindx).*Y{s}(testBindx,goodindx)));       % Covariance of non-cross prediction and data
                SSCc  = sum(sum(predY(:,goodindx).*Y{s}(testBindx,goodindx)));         % Covariance of cross prediction and data
                
                R.SN              = M.SN(m);
                R.params          = M.params{m};
                R.method{1, 1}    = M.method{m};
                R.trainMode{1, 1} = M.trainMode{m};
                R.xname{1, 1}     = M.xname{m};
                R.Rcv             = SSCc ./ sqrt(SSY.*SSP); % Double-Crossvalidated predictive correlation
                R.Rnc             = SSCn ./ sqrt(SSY.*SSP); % Not double-crossvalidated predictive correlation
                R.Ry              = SSCy ./ SSY;            % Reliability of data
                R.Rp              = SSCp ./ SSP;            % Reliability of prediction
                
                % If we knew the true pattern, the best correlation we
                % would expect is sqrt(Ry)
                % We also want to take into account the relaibility
                
                R.split = splits(sp);
                
                % Calucate Sparseness measures??????????
                Ws      = sort(abs(M.W{m}));
                Wss     = bsxfun(@rdivide,Ws,sum(Ws)); % standardized coefficients (to the sum overall
                R.spIdx = nanmean(Wss(end,:));         % Largest over the sum of the others
                
                N = size(Wss,1);
                w = (N-(1:N)'+ 0.5)/N;
                
                ginni    = 1 - 2*sum(bsxfun(@times, Wss, w));
                R.ginni  = nanmean(ginni); % Ginni index: mean over voxels
                %                 R.numReg = M.numReg(m); %% ?????????????????????
                
                RR = addstruct(RR,R);
            end % sp (?????????????????)
        end % m (models)
        varargout = {RR, Y{s}(testBindx,:), predY};
    case 'CONN:MDTB:evaluate_all'             % Evaluates predictive performance of connectivity model for all the subjects (or the group conn. model?!)
        % Example: sc1sc2_conn_model('CONN:MDTB:evaluate_all', 'method', 'plsr_1');
        
        sn         = returnSubjs;
        glm        = 7;              %% glmNum
        trainExper = 1;              %% Task set to train
        method     = 'plsr_1';       %% linRegress, ridgeFixed, plsr, pcr
        yname      = 'grey_nan';     %% cerebellar grey matter
        xname      = 'tesselsWB162'; %% cortical tessels used for modelling: tesselsWB362, tesselsWB642
        subset     = [];
        splitby    = [];
        
        vararginoptions(varargin, {'sn', 'glm', 'trainExper', 'method', ...
            'yname', 'xname', 'subest', 'splitby'});
        
        experStr = {'sc1', 'sc2'};
        Expers   = [1, 2];
        
        testExper = experStr{Expers ~= trainExper};
        
        % setting directories
        connModDir = fullfile(baseDir, experStr{trainExper}, connDir, sprintf('glm%d', glm), ...
            sprintf('mb4_%s_%s', xname, method));
        evalModDir = fullfile(baseDir, testExper, connDir, sprintf('glm%d', glm), ...
            sprintf('eval_mb4_%s_%s', xname, method));
        dircheck(evalModDir);
        
        EO = [];
        for s = sn
            fprintf('eval %s %s\n', method, subj_name{s});
            % load in the structure for the model
            M   = load(fullfile(connModDir, sprintf('mb4_%s_%s_%s.mat', xname, method, subj_name{s})));
            OUT = sc1sc2_conn_model('CONN:MDTB:evaluate', M, 'yname', yname, 'xname', xname, ...
                'subset', subset, 'splitby', splitby);
            outName = fullfile(evalModDir, sprintf('eval_mb4_%s_%s_%s', xname, method, subj_name{s}));
            save(outName,'-struct','OUT', '-v7.3');
            
            EO = addstruct(EO, OUT);
        end % s (sn)
        E_all_name = fullfile(evalModDir, sprintf('eval_mb4_%s_%s_all.mat', xname, method));
        save(E_all_name,'-struct','EO', '-v7.3');
    case 'evaluate_crossval_test'             % Checks whether the cross-session crossvalidation works appropriately
        % Borrowed from sc1sc2_connectivity
        % Example: sc1sc2_conn_model('evaluate_crossval_test')
        
        P1 = 20;   % Areas cortex
        P2 = 1000; % voxels cerebellum
        K  = 30;   % conditions
        
        sigma_s_1 = 1;    % Systematic signal in the cortex
        sigma_s_2 = 0;    % Additive signal in the cerebellum
        sigma_n_1 = 1;    % Systematic signal in the cortex
        sigma_n_2 = 1;    % Additive noise in the cerebellum
        sigma_w_s = 1;    % Connectivity weights for the signal
        sigma_w_n = 0;    % Connectivity weights for the noise
        
        Y.sess = kron([1;2],ones(K,1));
        Y.run = kron([1;2],ones(K,1));
        Y.cond = kron([1;1],[1:K]');
        
        T.SN=2;
        RR=[];
        for n=1:100
            WS = normrnd(0,sigma_w_s,P1,P2); WS(WS<0)=0; % non-negative connectivity weights for noise
            WN = normrnd(0,sigma_w_n,P1,P2); WN(WN<0)=0; % non-negative connectivity weights for signal
            
            N1  = normrnd(0,1,2*K,P1);
            N2  = N1*WN + normrnd(0,1,2*K,P2);
            S1  = normrnd(0,sigma_s_1,K,P1);
            S2  = S1*WS + normrnd(0,sigma_s_2,K,P2);
            S1=[S1;S1];
            S2=[S2;S2];
            
            T.WC{1} = (S1'*S1)\S1'*S2;  % Weights determined only on noise processes
            T.WC{1} = (N1'*N1)\N1'*N2;  % Weights determined only on noise processes
            T.lambda = 0;
            T.type = {[1]};
            T.typeNum = 1;
            
            Y.data = N2+S2;
            X.B    = N1+S1;
            R=sc1_imana_jd('encoding_crossval_evaluate',T,'Xdata',X,'Ydata',Y,'xname','other');
            RR=addstruct(RR,R);
        end;
        myboxplot([],[RR.Rcv RR.Rnc RR.R]);
        set(gca,'YLim',[-0.2 1]);
        drawline(0,'dir','horz');
        keyboard;
    case 'CONN:MDTB:evaluate_plot_method'     % Evaluation plots for a connectivity model
        % Borrowed from sc1sc2_connectivity
        % Example: sc1sc2_conn_model('CONN:MDTB:evaluate_plot_method', 'method', 'plsr_1');
        
        methods    = {'plsr_1'};     %% cell array containing all the models you want the plot for
        trainExper = 1;
        glm        = 7;
        xname      = 'tesselsWB162';
        yfield     = 'Rcv';
        xfield     = 'spIdx';
        
        vararginoptions(varargin, {'methods', 'trainExper',...
            'glm', 'xname', 'yfield', 'xfield'});
        
        experStr = {'sc1', 'sc2'};
        Expers   = [1, 2];
        
        testExper = experStr{Expers ~= trainExper};
        
        TT = [];
        for nm = 1:length(methods)
            % setting directories
            evalModDir = fullfile(baseDir, testExper, connDir, sprintf('glm%d', glm), ...
                sprintf('eval_mb4_%s_%s', xname, methods{nm}));
            name = fullfile(evalModDir, sprintf('eval_mb4_%s_%s_all.mat', xname, methods{nm}));
            T    = load(name);
            
            TT = addstruct(TT, T);
        end % nm (models)
        
        
        
        [methStr,~,TT.methNum] = unique(TT.method);
        [params,b,TT.parCat]=unique([TT.params],'rows');
        % Do an xyplot normalized to an upper noise ceiling, which only is
        % determined by the relability of the data
        %         xyplot(T.(xfield),T.(yfield)./sqrt(T.Ry),T.lamCat,'split',T.methNum,'style_thickline','leg',methStr,'subset',T.traindata == 2);
        %         xyplot(TT.(xfield),TT.(yfield)./sqrt(abs(TT.Ry)),TT.parCat,'split',TT.methNum,'style_thickline','leg',methStr);
        xyplot(TT.(xfield),TT.(yfield),TT.parCat,'split',TT.methNum,'style_thickline','leg',methStr);
        
        % Now determine a lower noise ceiling.... This is determined by the
        % reliability of the prediction, which is model dependent. We here
        % use the average across models
        noiseCeil = mean(T.Rp);
        drawline(noiseCeil,'dir','horz');
        set(gca,'YLim',[0.3 1]);
        set(gcf,'PaperPosition',[2 2 6 6]);
        
        varargout = {TT};
        % wysiwyg;
        
    case 'CONN:MDTB:PLSR:matlab_scratch'      % just practicing MATLAB's plsr
        % Example: sc1sc2_conn_model('CONN:MDTB:PLSR:matlab_scratch')
        
        sn             = returnSubjs;
        glm            = 7;
        experiment_num = 1;
        xres           = 162;
        atlas_res      = 32;
        
        vararginoptions(varargin, {'sn', 'glm', 'experiment_num', 'xres'});
        
        experiment = sprintf('sc%d', experiment_num);
        
        % setting directories
        betaDir = fullfile(baseDir, experiment, sprintf('Beta_GLM_%d', glm));
        glmDir  = fullfile(baseDir, experiment, sprintf('GLM_firstlevel_%d', glm));
        surfDir = fullfile(baseDir, experiment, sprintf('fs_LR_%d', atlas_res)); %% the directory where the atlas is located + giftti label files
        connDir = fullfile(baseDir, experiment, 'connModels');
        
        for s = sn
            
            % load in SPM_info
            T = load(fullfile(glmDir, subj_name{s}, 'SPM_info.mat'));
            
            % load in the beta file for each yeo_17WB
            load(fullfile(betaDir, subj_name{s}, sprintf('beta_regions_tesselsWB%d.mat', xres)));
            B1   = B;
            
            % load in the beta file for cerebellum_grey
            load(fullfile(betaDir, subj_name{s}, 'beta_regions_cerebellum_grey.mat'));
            %             load(fullfile(betaDir, subj_name{s}, 'beta_regions_Buckner_17.mat'));
            B2 = B;
            
            % get X and Y
            Xtmp = B1.mbetasUW(:, 1:end - length(runLst));
            % Y.
            %             Bcereb = cell2mat(B2.betasUW{15});
            Bcereb = cell2mat(B2.betasUW);
            Y1      = Bcereb';
            
            Y1 = Y1(:, 1:end - 16);
            Y1 = Y1';
            X = Xtmp';
            
            % for each run separately
            for r = 1:16
                Xf = X(T.run == r, :);
                Yf = Y1(T.run == r, :);
                Xi = zscore(Xf);
                Yi = zscore(Yf);
                
                %                 [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(Xi,Yi);
                
                out = conn_model_reg(Xi, Yi, 'model', 'plsr', 'nPC', 13);
                
                plot(1:size(PCTVAR, 2),cumsum(100*PCTVAR(2,:)),'-o');
                hold on;
            end
            
            keyboard
        end % s (s
        
    case 'CONN:MDTB:ols'                      % OLS regression for tessels, all MDTB
        % Example:sc1sc2_conn_model('CONN:MDTB:ols')
        % This is the simplest case where I just take the mean beta for
        % each cortical parcel (averaged across runs) and estimate the
        % connectivity.
        
        sn             = returnSubjs;
        glm            = 7;
        experiment_num = 1;
        xres           = 162; %% options: 162, 362, 642, 1002, 1442
        cv             = 1;   %% set to 1 or 0
        
        vararginoptions(varargin, {'sn', 'glm', 'experiment_num', 'xres', 'cv'});
        
        experiment = sprintf('sc%d', experiment_num);
        
        % pseudocode:
        % The multivariate multiple regression is:
        % Y = XW + E
        % Y(k-by-q), X(k-by-p), W(p-by-q).
        % where k is the number of conditions.
        % q is the number of cerebellar voxels
        % p is the number of cortical voxels.
        % load in the activity profiles of the perdictor and response
        % univariately prewhiten the activity profiles? Already done that!
        % Calculate the OLS estimates: What_OLS
        
        % setting directories
        betaDir    = fullfile(baseDir, experiment, sprintf('Beta_GLM_%d', glm));
        connModels = fullfile(baseDir, experiment, connDir, 'ols', sprintf('glm%d', glm), sprintf('tessels%d', xres));
        dircheck(connModels);
        glmDir  = fullfile(baseDir, experiment, sprintf('GLM_firstlevel_%d', glm));
        
        for s = sn
            tic;
            fprintf('subject: %s, ols\n', subj_name{s});
            subjConn    = [];
            tmp.sn      = s;
            subjConnDir = fullfile(connModels, subj_name{s});
            dircheck(subjConnDir);
            % load in the beta file for each yeo_17WB
            load(fullfile(betaDir, subj_name{s}, sprintf('beta_regions_tesselsWB%d.mat', xres)));
            B1   = B;
            
            % load in the beta file for cerebellum_grey
            load(fullfile(betaDir, subj_name{s}, 'beta_regions_cerebellum_grey.mat'));
            B2 = B;
            
            switch cv
                case 0 % no cross validation: use average across runs
                    % X.
                    % calculate mean across runs.
                    Xtmp = B1.mbetasUW(:, 1:end - length(runLst));
                    % reshape to get the runs on the 3rd dim
                    Xtmp1 = reshape(Xtmp, [size(Xtmp, 1), size(Xtmp, 2)/length(runLst), length(runLst)]);
                    Xtmp2 = nanmean(Xtmp1, 3);
                    X     = Xtmp2';
                    
                    % Y.
                    Bcereb = cell2mat(B2.mrbetasUW);
                    Y      = Bcereb';
                    
                    % adding the intercept
                    intercept = ones(size(X, 1), 1);
                    Xin       = [intercept, X];
                    
                    % 1. regression
                    What_ols = conn_model_reg(Xin, Y, 'model', 'ols');
                    
                    % calculate R2 and R
                    Yhat     = Xin*What_ols;
                    SSR      = sum(sum(sum((Y - Yhat).^2)));
                    SST      = sum(sum(sum(Y.^2)));
                    SSP      = sum(sum(sum(Yhat.^2)));
                    SSPA     = sum(sum(sum(Yhat.*Y)));
                    
                    R2   = 1 - SSR/SST;
                    R    = SSPA / sqrt(SST*SSP);
                    R2cv = NaN;
                    Rcv  = NaN;
                case 1 % use cross validation across runs
                    % load in SPM_info
                    T = load(fullfile(glmDir, subj_name{s}, 'SPM_info.mat'));
                    folds      = T.run;
                    conditions = T.cond;
                    %%% add 1 to the conditions if you want to include
                    %%% instructions too
                    conditions = conditions + 1;
                    % X.
                    X = B1.mbetasUW;
                    % Y.
                    Y = B2.betasUW{1}';
                    
                    % discard the intercepts
                    X = X(:, 1:end - length(runLst));
                    Y = Y(:, 1:end - length(runLst));
                    X = X';
                    Y = Y';
                    
                    [R2cv, Rcv, R2, R, What_ols] = conn_model_cv(X, Y, folds, conditions, 'model', 'ols');
                    
                    % cross validation
            end % cross validation?
            
            tmp.sn     = s;
            tmp.tessel = xres;
            tmp.W      = What_ols;
            tmp.R2     = R2;
            tmp.R      = R;
            tmp.R2cv   = R2cv;
            tmp.Rcv    = Rcv;
            
            fprintf('took %f\n', toc);
            
            subjConn = addstruct(subjConn, tmp);
            
            save(fullfile(subjConnDir, 'conn_ols.mat'), 'subjConn', '-v7.3');
            
            fprintf('Done\n\n');
        end % s (sn)
        varargout{1} = subjConn;
    case 'CONN:MDTB:rrr'                      % Reduced rank regression for tessels, all MDTB
        % Example:sc1sc2_conn_model('CONN:MDTB:rrr')
        
        % I'm not including options for rois. I am just using yeo_17WB and
        % cerebellum_grey.
        
        sn             = returnSubjs;
        glm            = 7;
        experiment_num = 1;
        rPC            = 17;
        xres           = 162; %% options: 162, 362, 642, 1002, 1442
        cv             = 1;   %% cv = 0 no cv, cv = 1 with cv
        
        
        vararginoptions(varargin, {'sn', 'glm', 'experiment_num', 'rPC', 'xres', 'cv'});
        
        experiment = sprintf('sc%d', experiment_num);
        
        % pseudocode:
        % The multivariate multiple regression is:
        % Y = XW + E
        % Y(k-by-q), X(k-by-p), W(p-by-q).
        % where k is the number of conditions.
        % q is the number of cerebellar voxels
        % p is the number of cortical voxels.
        % load in the activity profiles of the perdictor and response
        % univariately prewhiten the activity profiles? Already done that!
        % Calculate the OLS estimates: What_OLS
        % get the predicted values Yhat_OLS = XWhat_OLS
        % apply PCA to Yhat_OLS and retain the first r PCs (Ur)
        % Then What_RRR = What_OLS (UrUr').
        
        % setting directories
        betaDir = fullfile(baseDir, experiment, sprintf('Beta_GLM_%d', glm));
        connModels = fullfile(baseDir, experiment, connDir, 'rrr', sprintf('glm%d', glm), sprintf('tessels%d', xres));
        dircheck(connModels);
        glmDir  = fullfile(baseDir, experiment, sprintf('GLM_firstlevel_%d', glm));
        
        for s = sn
            tic;
            fprintf('subject: %s, rrr, rPc: %d\n', subj_name{s}, rPc);
            subjConn    = [];
            tmp.sn      = s;
            subjConnDir = fullfile(connModels, subj_name{s});
            dircheck(subjConnDir);
            % load in the beta file for each yeo_17WB
            load(fullfile(betaDir, subj_name{s}, sprintf('beta_regions_tesselsWB%d.mat', xres)));
            B1 = B;
            
            % load in the beta file for cerebellum_grey
            load(fullfile(betaDir, subj_name{s}, 'beta_regions_cerebellum_grey.mat'));
            B2 = B;
            
            switch cv
                case 0 % no cross validation: use average across runs
                    % X.
                    % calculate mean across runs.
                    Xtmp = B1.mbetasUW(:, 1:end - length(runLst));
                    % reshape to get the runs on the 3rd dim
                    Xtmp1 = reshape(Xtmp, [size(Xtmp, 1), size(Xtmp, 2)/length(runLst), length(runLst)]);
                    Xtmp2 = nanmean(Xtmp1, 3);
                    X     = Xtmp2';
                    
                    % Y.
                    Bcereb = cell2mat(B2.mrbetasUW);
                    Y = Bcereb';
                    
                    % adding the intercept
                    intercept = ones(size(X, 1), 1);
                    Xin       = [intercept, X];
                    
                    % 1. regression
                    What_rrr = conn_model_reg(Xin, Y, 'model', 'rrr', 'nPC', rPC);
                    
                    % 2. calculate R2
                    Yhat     = Xin*What_rrr;
                    SSR      = sum(sum(sum((Y - Yhat).^2)));
                    SST      = sum(sum(sum(Y.^2)));
                    SSP      = sum(sum(sum(Yhat.^2)));
                    SSPA     = sum(sum(sum(Yhat.*Y)));
                    
                    R2   = 1 - SSR/SST;
                    R    = SSPA / sqrt(SST*SSP);
                    R2cv = NaN;
                    Rcv  = NaN;
                case 1 % use cross validation across runs
                    % load in SPM_info
                    T = load(fullfile(glmDir, subj_name{s}, 'SPM_info.mat'));
                    folds      = T.run;
                    conditions = T.cond;
                    %%% add 1 to the conditions if you want to include
                    %%% instructions too
                    conditions = conditions + 1;
                    % X.
                    X = B1.mbetasUW;
                    % Y.
                    Y = B2.betasUW{1}';
                    
                    % discard the intercepts
                    X = X(:, 1:end - length(runLst));
                    Y = Y(:, 1:end - length(runLst));
                    X = X';
                    Y = Y';
                    
                    [R2cv, Rcv, R2, R, What_rrr] = conn_model_cv(X, Y, folds, conditions, 'model', 'rrr', 'nPC', rPC);
                    
            end % cross validation?
            
            tmp.sn     = s;
            tmp.tessel = xres;
            tmp.W      = What_rrr;
            tmp.rPc    = rPc;
            tmp.R2     = R2;
            tmp.R      = R;
            tmp.R2cv   = R2cv;
            tmp.Rcv    = Rcv;
            
            fprintf('took %f\n', toc);
            
            subjConn = addstruct(subjConn, tmp);
            
            save(fullfile(subjConnDir, sprintf('conn_rrr_%d.mat', rPc)), 'subjConn', '-v7.3');
            
            fprintf('Done\n\n');
        end % s (sn)
        varargout{1} = subjConn;
    case 'CONN:MDTB:ridge'                    % Ridge regression for tessels, all MDTB
        % Example:sc1sc2_conn_model('CONN:MDTB:ridge')
        
        % I'm not including options for rois. I am just using yeo_17WB and
        % cerebellum_grey.
        
        sn             = returnSubjs;
        glm            = 7;
        experiment_num = 1;
        xres           = 162;
        lambda         = 0.01; %% options: 162, 362, 642, 1002, 1442
        cv             = 1;    %% cv = 0 no cv, cv = 1 with cv
        
        
        vararginoptions(varargin, {'sn', 'glm', 'experiment_num', 'xres', 'lambda', 'cv'});
        
        experiment = sprintf('sc%d', experiment_num);
        
        % pseudocode:
        % What = (X'X + LambdaI)X'Y
        
        % setting directories
        betaDir = fullfile(baseDir, experiment, sprintf('Beta_GLM_%d', glm));
        connModels = fullfile(baseDir, experiment, connDir, 'ridge', sprintf('glm%d', glm), sprintf('tessels%d', xres));
        dircheck(connModels);
        glmDir  = fullfile(baseDir, experiment, sprintf('GLM_firstlevel_%d', glm));
        
        for s = sn
            tic;
            fprintf('subject: %s, Ridge, lambda: %0.4f\n', subj_name{s}, lambda);
            subjConn    = [];
            tmp.sn      = s;
            subjConnDir = fullfile(connModels, subj_name{s});
            dircheck(subjConnDir);
            % load in the beta file for each yeo_17WB
            load(fullfile(betaDir, subj_name{s}, sprintf('beta_regions_tesselsWB%d.mat', xres)));
            B1 = B;
            
            % load in the beta file for cerebellum_grey
            load(fullfile(betaDir, subj_name{s}, 'beta_regions_cerebellum_grey.mat'));
            B2 = B;
            
            switch cv
                case 0 % no cross validation: use average across runs
                    % X.
                    % calculate mean across runs.
                    Xtmp = B1.mbetasUW(:, 1:end - length(runLst));
                    % reshape to get the runs on the 3rd dim
                    Xtmp1 = reshape(Xtmp, [size(Xtmp, 1), size(Xtmp, 2)/length(runLst), length(runLst)]);
                    Xtmp2 = nanmean(Xtmp1, 3);
                    X     = Xtmp2';
                    
                    % Y.
                    Bcereb = cell2mat(B2.mrbetasUW);
                    Y = Bcereb';
                    % 1.mean center the inputs
                    Xc = bsxfun(@minus, X, nanmean(X));
                    Yc = bsxfun(@minus, Y, nanmean(Y));
                    
                    % 1. regression
                    What_ridge = conn_model_reg(Xc, Yc, 'model', 'ridge', 'lambda', lambda);
                    
                    % 2. calculate R2
                    Yhat     = Xc*What_ridge;
                    SSR      = sum(sum(sum((Yc - Yhat).^2)));
                    SST      = sum(sum(sum(Y.^2)));
                    SSP      = sum(sum(sum(Yhat.^2)));
                    SSPA     = sum(sum(sum(Yhat.*Y)));
                    
                    R2   = 1 - SSR/SST;
                    R    = SSPA / sqrt(SST*SSP);
                    R2cv = NaN;
                    Rcv  = NaN;
                case 1 % use cross validation across runs
                    % load in SPM_info
                    T = load(fullfile(glmDir, subj_name{s}, 'SPM_info.mat'));
                    folds      = T.run;
                    conditions = T.cond;
                    %%% add 1 to the conditions if you want to include
                    %%% instructions too
                    conditions = conditions + 1;
                    % X.
                    X = B1.mbetasUW;
                    % Y.
                    Y = B2.betasUW{1}';
                    
                    % discard the intercepts
                    X = X(:, 1:end - length(runLst));
                    Y = Y(:, 1:end - length(runLst));
                    X = X';
                    Y = Y';
                    
                    [R2cv, Rcv, R2, R, What_ridge] = conn_model_cv(X, Y, folds, conditions, 'model', 'ridge', 'lambda', lambda);
                    
            end % cross validation?
            
            tmp.sn     = s;
            tmp.tessel = xres;
            tmp.W      = What_ridge;
            tmp.lambda = lambda;
            tmp.R2     = R2;
            tmp.R      = R;
            tmp.R2cv   = R2cv;
            tmp.Rcv    = Rcv;
            
            fprintf('took %f\n', toc);
            
            subjConn = addstruct(subjConn, tmp);
            
            save(fullfile(subjConnDir, sprintf('conn_ridge_%0.4f.mat', lambda)), 'subjConn', '-v7.3');
            
            fprintf('Done\n\n');
        end % s (sn)
        varargout{1} = subjConn;
    case 'CONN:MDTB:plsr_matlab'              % Partial Least squarese regression, matlab, all MDTB
        % Example: sc1sc2_conn_model('CONN:MDTB:plsr_matlab')
        
        sn             = returnSubjs;
        glm            = 7;
        experiment_num = 1;
        rPC            = 0;
        xres           = 162; %% options: 162, 362, 642, 1002, 1442
        cv             = 1;   %% cv = 0 no cv, cv = 1 with cv
        atlas_res      = 32;
        
        vararginoptions(varargin, {'sn', 'glm', 'experiment_num', 'rPC', 'xres', 'cv'});
        
        experiment = sprintf('sc%d', experiment_num);
        
        % setting directories
        betaDir = fullfile(baseDir, experiment, sprintf('Beta_GLM_%d', glm));
        connModels = fullfile(baseDir, experiment, connDir, 'plsr_matlab', sprintf('glm%d', glm), sprintf('tessels%d', xres));
        dircheck(connModels);
        glmDir  = fullfile(baseDir, experiment, sprintf('GLM_firstlevel_%d', glm));
        
        for s = sn
            tic;
            fprintf('subject: %s, plsr, rPc: %d\n', subj_name{s}, rPc);
            subjConn    = [];
            tmp.sn      = s;
            subjConnDir = fullfile(connModels, subj_name{s});
            dircheck(subjConnDir);
            % load in the beta file for each yeo_17WB
            load(fullfile(betaDir, subj_name{s}, sprintf('beta_regions_tesselsWB%d.mat', xres)));
            B1 = B;
            
            % load in the beta file for cerebellum_grey
            load(fullfile(betaDir, subj_name{s}, 'beta_regions_cerebellum_grey.mat'));
            B2 = B;
            
            switch cv
                case 0 % no cross validation: use average across runs
                    % X.
                    % calculate mean across runs.
                    Xtmp = B1.mbetasUW(:, 1:end - length(runLst));
                    % reshape to get the runs on the 3rd dim
                    Xtmp1 = reshape(Xtmp, [size(Xtmp, 1), size(Xtmp, 2)/length(runLst), length(runLst)]);
                    Xtmp2 = nanmean(Xtmp1, 3);
                    X     = Xtmp2';
                    
                    % Y.
                    Bcereb = cell2mat(B2.mrbetasUW);
                    Y = Bcereb';
                    
                    % adding the intercept
                    intercept = ones(size(X, 1), 1);
                    Xin       = [intercept, X];
                    
                    % 1. regression
                    out_plsr = conn_model_reg(Xin, Y, 'model', 'plsr_matlab', 'nPC', rPC);
                    
                    What_plsr = out_plsr.BETA;
                    % 2. calculate R2
                    Yhat     = Y - out_plsr.stats.Yresiduals;
                    SSR      = sum(sum(sum((out_plsr.stats.Yresiduals).^2)));
                    SST      = sum(sum(sum(Y.^2)));
                    SSP      = sum(sum(sum(Yhat.^2)));
                    SSPA     = sum(sum(sum(Yhat.*Y)));
                    
                    R2   = 1 - SSR/SST;
                    R    = SSPA / sqrt(SST*SSP);
                    R2cv = NaN;
                    Rcv  = NaN;
                case 1 % use cross validation across runs
                    % load in SPM_info
                    T = load(fullfile(glmDir, subj_name{s}, 'SPM_info.mat'));
                    folds      = T.run;
                    conditions = T.cond;
                    %%% add 1 to the conditions if you want to include
                    %%% instructions too
                    conditions = conditions + 1;
                    % X.
                    X = B1.mbetasUW;
                    % Y.
                    Y = B2.betasUW{1}';
                    
                    % discard the intercepts
                    X = X(:, 1:end - length(runLst));
                    Y = Y(:, 1:end - length(runLst));
                    X = X';
                    Y = Y';
                    
                    [R2cv, Rcv, R2, R, What_plsr] = conn_model_cv(X, Y, folds, conditions, 'model', 'plsr_matlab', 'nPC', rPC);
                    
            end % cross validation?
            
            tmp.sn     = s;
            tmp.tessel = xres;
            tmp.W      = What_plsr;
            tmp.rPc    = rPc;
            tmp.R2     = R2;
            tmp.R      = R;
            tmp.R2cv   = R2cv;
            tmp.Rcv    = Rcv;
            
            fprintf('took %f\n', toc);
            
            subjConn = addstruct(subjConn, tmp);
            
            save(fullfile(subjConnDir, sprintf('conn_plsr_matlab_%d.mat', rPc)), 'subjConn', '-v7.3');
            
            fprintf('Done\n\n');
        end % s (sn)
    case 'CONN:MDTB:plsr'                     % Partial Least squares regression, all MDTB
        % Example: sc1sc2_conn_model('CONN:MDTB:plsr')
        
        sn             = returnSubjs;
        glm            = 7;
        experiment_num = 1;
        rPC            = 0;
        xres           = 162; %% options: 162, 362, 642, 1002, 1442
        cv             = 1;   %% cv = 0 no cv, cv = 1 with cv
        atlas_res      = 32;
        
        vararginoptions(varargin, {'sn', 'glm', 'experiment_num', 'rPC', 'xres', 'cv'});
        
        experiment = sprintf('sc%d', experiment_num);
        
        % setting directories
        betaDir = fullfile(baseDir, experiment, sprintf('Beta_GLM_%d', glm));
        connModels = fullfile(baseDir, experiment, connDir, 'plsr_matlab', sprintf('glm%d', glm), sprintf('tessels%d', xres));
        dircheck(connModels);
        glmDir  = fullfile(baseDir, experiment, sprintf('GLM_firstlevel_%d', glm));
        
        for s = sn
            tic;
            fprintf('subject: %s, plsr, rPc: %d\n', subj_name{s}, rPc);
            subjConn    = [];
            tmp.sn      = s;
            subjConnDir = fullfile(connModels, subj_name{s});
            dircheck(subjConnDir);
            % load in the beta file for each yeo_17WB
            load(fullfile(betaDir, subj_name{s}, sprintf('beta_regions_tesselsWB%d.mat', xres)));
            B1 = B;
            
            % load in the beta file for cerebellum_grey
            load(fullfile(betaDir, subj_name{s}, 'beta_regions_cerebellum_grey.mat'));
            B2 = B;
            
            switch cv
                case 0 % no cross validation: use average across runs
                    % X.
                    % calculate mean across runs.
                    Xtmp = B1.mbetasUW(:, 1:end - length(runLst));
                    % reshape to get the runs on the 3rd dim
                    Xtmp1 = reshape(Xtmp, [size(Xtmp, 1), size(Xtmp, 2)/length(runLst), length(runLst)]);
                    Xtmp2 = nanmean(Xtmp1, 3);
                    X     = Xtmp2';
                    
                    % Y.
                    Bcereb = cell2mat(B2.mrbetasUW);
                    Y = Bcereb';
                    
                    % adding the intercept
                    intercept = ones(size(X, 1), 1);
                    Xin       = [intercept, X];
                    
                    % 1. regression
                    out_plsr = conn_model_reg(Xin, Y, 'model', 'plsr');
                    
                    What_plsr = out_plsr.Wpls;
                    % 2. calculate R2
                    Yhat     = out_plsr.Yhat;
                    SSR      = sum(sum(sum((Y - Yhat).^2)));
                    SST      = sum(sum(sum(Y.^2)));
                    SSP      = sum(sum(sum(Yhat.^2)));
                    SSPA     = sum(sum(sum(Yhat.*Y)));
                    
                    R2   = 1 - SSR/SST;
                    R    = SSPA / sqrt(SST*SSP);
                    R2cv = NaN;
                    Rcv  = NaN;
                case 1 % use cross validation across runs
                    % load in SPM_info
                    T = load(fullfile(glmDir, subj_name{s}, 'SPM_info.mat'));
                    folds      = T.run;
                    conditions = T.cond;
                    %%% add 1 to the conditions if you want to include
                    %%% instructions too
                    conditions = conditions + 1;
                    % X.
                    X = B1.mbetasUW;
                    % Y.
                    Y = B2.betasUW{1}';
                    
                    % discard the intercepts
                    X = X(:, 1:end - length(runLst));
                    Y = Y(:, 1:end - length(runLst));
                    X = X';
                    Y = Y';
                    
                    [R2cv, Rcv, R2, R, What_plsr] = conn_model_cv(X, Y, folds, conditions, 'model', 'plsr');
                    
            end % cross validation?
            
            tmp.sn     = s;
            tmp.tessel = xres;
            tmp.W      = What_plsr;
            tmp.rPc    = rPc;
            tmp.R2     = R2;
            tmp.R      = R;
            tmp.R2cv   = R2cv;
            tmp.Rcv    = Rcv;
            
            fprintf('took %f\n', toc);
            
            subjConn = addstruct(subjConn, tmp);
            
            save(fullfile(subjConnDir, sprintf('conn_plsr_%d.mat', rPc)), 'subjConn', '-v7.3');
            
            fprintf('Done\n\n');
        end % s (sn)
        
    case 'CONN:MDTB:ridge_lamb'               % Ridge regression with different lambdas_testing
        % Example: sc1sc2_conn_model('CONN:MDTB:ridge_lamb', 'sn', [2,3,4,6,8,9,10,12,14,15,17])
        
        sn     = returnSubjs;
        xres   = 162; %% options: 162, 362, 642, 1002, 1442
        
        vararginoptions(varargin, {'sn', 'xres'});
        
        lambdaArr1 = 0.1:0.1:1;
        lambdaArr2 = 1:20;
        for l1 = lambdaArr1
            sc1sc2_conn_model('CONN:MDTB:ridge', 'sn', sn, 'xres', xres, 'cv', 1, 'lambda', l1);
        end % l (lambdas)
        
        for l2 = lambdaArr2
            sc1sc2_conn_model('CONN:MDTB:ridge', 'sn', sn, 'xres', xres, 'cv', 1, 'lambda', l2);
        end % l (lambdas)
    case 'CONN:MDTB:rrr_Pc'                   % Reduced rank regression with different rPcs_testing
        % Example: sc1sc2_conn_model('CONN:MDTB:rrr_Pc', 'sn', [2,3,4,6,8,9,10,12,14,15,17])
        
        sn     = returnSubjs;
        xres   = 162; %% options: 162, 362, 642, 1002, 1442
        
        vararginoptions(varargin, {'sn', 'xres'});
        
        mxPC = 45;
        PcArr = 1:mxPC;
        for p = PcArr
            sc1sc2_conn_model('CONN:MDTB:rrr', 'sn', sn, 'xres', xres, 'cv', 1, 'rPc', p);
        end % p (PCs)
    case 'CONN:MDTB:plsr_matlab_Pc'           % PLS regression with different rPcs_testing
        % Example: sc1sc2_conn_model('CONN:MDTB:plsr_matlab_Pc', 'sn', [2,3,4,6,8,9,10,12,14,15,17])
        
        sn     = returnSubjs;
        xres   = 162; %% options: 162, 362, 642, 1002, 1442
        
        vararginoptions(varargin, {'sn', 'xres'});
        
        mxPC  = 45;
        PcArr = 1:mxPC;
        for p = PcArr
            sc1sc2_conn_model('CONN:MDTB:plsr_matlab', 'sn', sn, 'xres', xres, 'cv', 1, 'rPc', p);
        end % p (PCs)
    case 'CONN:MDTB:ridge_lamb_all'           % Ridge regression with different lambdas_testing for different tesssels
        % Example: sc1sc2_conn_model('CONN:MDTB:ridge_lamb_all', 'sn', [2,3,4,6,8,9,10,12,14,15,17], 'xresArr', [162, 362, 642])
        
        sn        = returnSubjs;
        xresArr   = [162, 362, 642];
        
        vararginoptions(varargin, {'sn', 'xresArr'});
        
        for nn = xresArr
            sc1sc2_conn_model('CONN:MDTB:ridge_lamb', 'sn', sn, 'xres', nn);
        end % nn (xres)
    case 'CONN:MDTB:rrr_Pc_all'               % Reduced rank regression with different rPcs_testing for different tesssels
        % Example: sc1sc2_conn_model('CONN:MDTB:rrr_Pc_all', 'sn', [2,3,4,6,8,9,10,12,14,15,17], 'xresArr', [162, 362, 642])
        
        sn        = returnSubjs;
        xresArr   = [162, 362, 642];
        
        vararginoptions(varargin, {'sn', 'xresArr'});
        
        for nn = xresArr
            sc1sc2_conn_model('CONN:MDTB:rrr_Pc', 'sn', sn, 'xres', nn);
        end % nn (xres)
    case 'CONN:MDTB:plsr_matlab_Pc_all'       % PLS regression with different rPcs_testing for different tesssels
        % Example: sc1sc2_conn_model('CONN:MDTB:plsr_matlab_Pc_all', 'sn', [2,3,4,6,8,9,10,12,14,15,17], 'xresArr', [162, 362, 642])
        
        sn        = returnSubjs;
        xresArr   = [162, 362, 642];
        
        vararginoptions(varargin, {'sn', 'xresArr'});
        
        for nn = xresArr
            sc1sc2_conn_model('CONN:MDTB:plsr_matlab_Pc' , 'sn', sn, 'xres', nn);
        end % nn (xres)
    case 'CONN:MDTB:model_df'                 % creates a dataframe with data for all the subjects
        % Example: sc1sc2_conn_model('CONN:MDTB:model_df' , 'model', 'ridge', 'xresArr', [162])
        sn             = returnSubjs;
        experiment_num = 1;
        glm            = 7;
        xres           = 162;
        model          = 'ridge';
        
        vararginoptions(varargin, {'sn', 'experiment_num', 'glm', 'xres', 'model'})
        
        experiment = sprintf('sc%d', experiment_num);
        
        connDir = fullfile(baseDir, experiment, 'connModels', model, sprintf('glm%d', glm));
        
        df = []; % dataframe with all the data for all the subjects
        for s = sn
            % loads in the R2cv values for the model
            fprintf('doing tessel %d, subject: %s\n', xres, subj_name{s});
            files = dir(fullfile(connDir, sprintf('tessels%d', xres), subj_name{s}, 'conn_*'));
            for f = 1:length(files)
                load(fullfile(files(f).folder, files(f).name));
                tmp = subjConn;
                tmp = rmfield(tmp, 'W');
                df = addstruct(df, tmp);
            end % f (files)
        end % s (sn)
        save(fullfile(connDir, sprintf('ridgeModel_%d.mat', xres)), 'df', '-v7.3');
    case 'CONN:MDTB:model_test'               % selectting the best model for ridge regression and rrr
        % plots R2cv vs Lambda for ridge regression and R2cv vs rPc for rrr
        % Example: sc1sc2_conn_model('CONN:MDTB:model_test', 'model', 'ridge')
        
        sn             = returnSubjs;
        experiment_num = 1;
        glm            = 7;
        xresArr        = [162, 362, 642];
        model          = 'ridge';
        yvar           = 'R2cv';
        
        vararginoptions(varargin, {'sn', 'experiment_num', 'glm', 'xresArr', 'model'})
        
        experiment = sprintf('sc%d', experiment_num);
        
        connDir = fullfile(baseDir, experiment, 'connModels', model, sprintf('glm%d', glm));
        
        switch model
            case 'rrr'
                xvar = 'rPc';
            case 'ridge'
                xvar = 'lambda';
        end
        
        for nn = xresArr
            % load in the dataframe with model selection info for all the
            % subjects
            load(fullfile(connDir, sprintf('%sModel_%d.mat', model, nn)));
            %             xyplot(df.(xvar),df.(yvar), [], 'CAT', CAT, 'split', txx.(which), 'label', taskNames_Inst);
            
        end % nn (xres)
        
    case 'Houskeeping:renameSPM'              % Changes directory paths in SPM struct
        % rename SPM directories
        % Example: sc1sc2_conn_model('Houskeeping:renameSPM', 'experiment_num', 2, 'glm', 8, 'sn', 6)
        
        sn             = returnSubjs;
        experiment_num = 2;
        glm            = 7;
        
        vararginoptions(varargin, {'sn', 'experiment_num', 'glm'});
        
        experiment = sprintf('sc%d', experiment_num);
        
        glmDir     = fullfile(baseDir, experiment, sprintf('GLM_firstlevel_%d', glm));
        imagingDir = fullfile(baseDir, 'sc1/imaging_data'); %% all the imaging files are in sc1
        
        for s = sn
            fprintf('changing directories for %s \n', subj_name{s});
            newGLMDir   = fullfile(glmDir,subj_name{s});
            newRawDir   = fullfile(imagingDir,subj_name{s});
            
            % load SPM file
            if exist(fullfile(newGLMDir, 'SPM.mat'), 'file')
                load(fullfile(newGLMDir, 'SPM.mat'));
            else
                load(fullfile(newGLMDir, 'SPM_light.mat'));
            end
            
            
            SPM         = spmj_move_rawdata(SPM,newRawDir);
            SPM.swd     = fullfile(newGLMDir);
            save(fullfile(newGLMDir,'SPM.mat'),'SPM','-v7.3');
            varargout{1} = SPM;
        end % s (sn)
    case 'Houskeeping:move_files'             % moving files
        % Example: sc1sc2_conn_model('Houskeeping:move_files', 'experiment_num', 2, 'copywhich', 'SPM_info')
        
        sn             = returnSubjs;
        experiment_num = 1;
        glm            = 7;
        copywhich      = 'GLM';
        serverDir      = 'smb://129.100.119.43/motorcontrol/data/super_cerebellum_new';
        roi_name       = 'cerebellum_grey'; % options: cerebellum_grey, 'tesselsWB162';
        
        vararginoptions(varargin, {'sn', 'glm', 'experiment_num', 'con_vs', 'nTrans', 'copywhich', 'roi_name'});
        
        switch copywhich
            case 'region_cortex'
                experiment = sprintf('sc%d', experiment_num);
                sourceDir  = fullfile(serverDir, experiment, 'RegionOfInterest', 'data');
                destDir    = fullfile(baseDir, experiment, 'RegionOfInterest', 'data');
                
                for s = sn
                    sourceFile  = fullfile(sourceDir, subj_name{s}, 'regions_cortex.mat');
                    destDirSubj = fullfile(destDir, subj_name{s});
                    [success(s), Message{s}, ~] = copyfile(sourceFile, destDirSubj);
                end % sn
            case 'SPM_info'
                experiment = sprintf('sc%d', experiment_num);
                sourceDir  = fullfile(serverDir, experiment, sprintf('GLM_firstlevel_%d', glm));
                destDir    = fullfile(baseDir, experiment, sprintf('GLM_firstlevel_%d', glm));
                
                for s = sn
                    sourceFolder = fullfile(sourceDir, subj_name{s});
                    destFolder   = fullfile(destDir, subj_name{s});
                    dircheck(destFolder);
                    cd(sourceFolder);
                    %                     system(sprintf('find -name -exec cp  -R %s %s;', sourceFolder, destFolder));
                    [success(s), message{s}, ~] = copyfile(fullfile(sourceFolder, 'SPM_info.mat'), destFolder, 'f');
                    
                    if success(s) == 1
                        fprintf('%s coppied from the server\n', subj_name{s});
                    else
                        fprintf('copying %s from the server failed\n', subj_name{s});
                        fprintf('%s\n', message{s});
                    end
                end % sn
            case 'SPM_light'
                experiment = sprintf('sc%d', experiment_num);
                sourceDir  = fullfile(serverDir, experiment, sprintf('GLM_firstlevel_%d', glm));
                destDir    = fullfile(baseDir, experiment, sprintf('GLM_firstlevel_%d', glm));
                
                for s = sn
                    sourceFolder = fullfile(sourceDir, subj_name{s});
                    destFolder   = fullfile(destDir, subj_name{s});
                    dircheck(destFolder);
                    cd(sourceFolder);
                    %                     system(sprintf('find -name -exec cp  -R %s %s;', sourceFolder, destFolder));
                    %                     [success(s), message{s}, ~] = copyfile(fullfile(sourceFolder, 'SPM_light.mat'), destFolder, 'f');
                    [success(s), message{s}, ~] = copyfile(fullfile(sourceFolder, 'SPM.mat'), destFolder, 'f');
                    
                    if success(s) == 1
                        fprintf('%s coppied from the server\n', subj_name{s});
                    else
                        fprintf('copying %s from the server failed\n', subj_name{s});
                        fprintf('%s\n', message{s});
                    end
                end % sn
            case 'beta'
                experiment = sprintf('sc%d', experiment_num);
                sourceDir  = fullfile(serverDir, experiment, sprintf('GLM_firstlevel_%d', glm));
                destDir    = fullfile(baseDir, experiment, sprintf('GLM_firstlevel_%d', glm));
                
                for s = sn
                    sourceFolder = fullfile(sourceDir, subj_name{s});
                    destFolder   = fullfile(destDir, subj_name{s});
                    dircheck(destFolder);
                    cd(sourceFolder);
                    
                    betas = dir('beta_*');
                    
                    for ib = 1:length(betas)
                        [success(s), message{s}, ~] = copyfile(fullfile(sourceFolder, betas(ib).name), destFolder, 'f');
                        
                        if success(s) == 1
                            fprintf('%s coppied %s from the server\n', betas(ib).name, subj_name{s});
                        else
                            fprintf('copying %s from the server failed\n', subj_name{s});
                            fprintf('%s\n', message{s});
                        end
                        
                    end % ib
                end % sn
            case 'others'
                experiment = sprintf('sc%d', experiment_num);
                sourceDir  = fullfile(serverDir, experiment, sprintf('GLM_firstlevel_%d', glm));
                destDir    = fullfile(baseDir, experiment, sprintf('GLM_firstlevel_%d', glm));
                
                for s = sn
                    sourceFolder = fullfile(sourceDir, subj_name{s});
                    destFolder   = fullfile(destDir, subj_name{s});
                    dircheck(destFolder);
                    cd(sourceFolder);
                    
                    [success(s), message{s}, ~] = copyfile(fullfile(sourceFolder, 'RPV.nii'), destFolder, 'f');
                    
                    if success(s) == 1
                        fprintf('RPV for %s coppied from the server\n', subj_name{s});
                    else
                        fprintf('copying RPV for %s from the server failed\n', subj_name{s});
                        fprintf('%s\n', message{s});
                    end
                    
                    [success(s), message{s}, ~] = copyfile(fullfile(sourceFolder, 'ResMS.nii'), destFolder, 'f');
                    
                    if success(s) == 1
                        fprintf('ResMS for %s coppied from the server\n', subj_name{s});
                    else
                        fprintf('copying ResMS for %s from the server failed\n', subj_name{s});
                        fprintf('%s\n', message{s});
                    end
                    
                    [success(s), message{s}, ~] = copyfile(fullfile(sourceFolder, 'mask.nii'), destFolder, 'f');
                    
                    if success(s) == 1
                        fprintf('mask for %s coppied from the server\n', subj_name{s});
                    else
                        fprintf('copying mask for %s from the server failed\n', subj_name{s});
                        fprintf('%s\n', message{s});
                    end
                end % sn
            case 'beta_roi'
                experiment = sprintf('sc%d', experiment_num);
                destDir   = fullfile(serverDir, experiment, 'beta_roi', sprintf('glm%d', glm));
                sourceDir = fullfile(baseDir, experiment, 'beta_roi', sprintf('glm%d', glm));
                for s = sn
                    sourceFolder = fullfile(sourceDir, subj_name{s});
                    destFolder   = fullfile(destDir, subj_name{s});
                    dircheck(destFolder);
                    
                    
                    file2copy = fullfile(sourceFolder, sprintf('Y_info_glm%d_%s.mat', glm, roi_name));
                    
                    [success(s), message{s}, ~] = copyfile(file2copy, destFolder, 'f');
                    
                    if success(s) == 1
                        fprintf('cerebellum grey Y info for %s coppied to the server\n', subj_name{s});
                    else
                        fprintf('copying cerebellum grey Y info for %s to the server failed\n', subj_name{s});
                        fprintf('%s\n', message{s});
                    end
                end % sn
                
        end
        
    case 'CONN:TOY'                           % Applies different models to toy examples: for comparison
        % Example: sc1sc2_conn_model('CONN:TOY', 'model', 'PLS-R1')
        
        model = 'PCR1'; % other option is PLS-R
        
        vararginoptions(varargin, {'model'});
        
        
        switch model
            case 'PCR1'   % Principal component Regression implementation 1
                % 1. apply PCA to X0:
                [V, Z] = pca(X0); % Z scores and V loadings
                
                % 2. regress Y on score
                W = (Z'*Z)\(Z'*Y);
            case 'PCR2'   % Principal component Regression implementation 2
            case 'PLS-R1' % PLS-R 1 implementation
                % just checking steps of PLS-R with an example from
                % (Krishnan et al., 2011)
                X1 = [2 5 6 1 9 1 7 6 2 1 7 3; ...
                    4 1 5 8 8 7 2 8 6 4 8 2; ...
                    5 8 7 3 7 1 7 4 5 1 4 3];
                X2 = [3 3 7 6 1 1 10 2 2 1 7 4; ...
                    2 3 8 7 1 6 9 1 8 8 1 6;  ...
                    1 7 3 1 1 3 1 8 1 3 9 5];
                X3 = [9 0 7 1 8 7 4 2 3 6 2 7;  ...
                    8 0 6 5 9 7 4 4 2 10 3 8; ...
                    7 7 4 5 7 6 7 6 5 4 8 8];
                X = [X1; X2; X3];
                
                Y1 = [15 600; 19 520; 18 545];
                Y2 = [22 426; 21 404; 23 411];
                Y3 = [29 326; 30 309; 30 303];
                
                Y = [Y1; Y2; Y3];
                
                % z-score X and Y
                X0 = zscore(X);
                Y0 = zscore(Y);
                
                iterate = true;
                Xi = X0;
                Yi = Y0;
                
                i = 1;
                L = rank(X0);
                
                T = []; % X scores
                U = []; % Y scores
                P = []; % X loadings
                W = []; % X weights
                C = []; % Y weights
                Q = [];
                b = []; % regression coefficients
                
                while iterate
                    % inspecting just one iteration
                    %%% matrix of covariance
                    Ri = Xi'*Yi;
                    
                    %%% SVD of covariance matrix
                    [Wi, D, Ci] = svd(Ri);
                    
                    %%% the first X score
                    wi = Wi(:, 1);
                    ci = Ci(:, 1);
                    ti = (Xi*wi);
                    ti = ti/sqrt(ti'*ti);
                    
                    %%% The loading of X0 on ti
                    pi = Xi'*ti;
                    
                    
                    %%% ith latent variable of Y
                    ui = Yi*ci;
                    
                    %%% reconstruct X and Y
                    Yihat1 = ui*ci';
                    Xihat = ti*pi';
                    
                    %%% estimate bi
                    bi = ti'*ui;
                    
                    Yihat = ti*bi*ci';
                    
                    % update T, U, P, W, C, b
                    T = [T, ti];
                    U = [U, ui];
                    P = [P, pi];
                    Q = [];
                    W = [W, wi];
                    C = [C, ci];
                    b = [b, bi];
                    
                    i = i + 1;
                    
                    % deflation
                    Xi = Xi - Xihat;
                    Yi = Yi - Yihat;
                    
                    if size(W, 2) == L
                        iterate = false;
                    end
                end % iterate
                
                
                [Pm, Qm, Tm, Um, Wpls_m,PCTVARm,MSEm,stats_m] = plsregress(X0,Y0);
                
                
                keyboard;
                Wpls1 = W*inv(P'*W)*C';
                D = (T'*T)\(T'*U);
                Wpls2 = pinv(P')*D*C';
                B = diag(b);
                Wpls3 = pinv(P')*B*C';
                keyboard;
                
        end
    case 'CONN:TOY_plsr'                      % playing around with matlab's plsregress
        % Example: sc1sc2_conn_model('CONN:TOY_plsr');
        
        load spectra
        X = NIR;
        y = octane;
        
        [XL,yl,XS,YS,beta,PCTVAR] = plsregress(X,y,10);
        
        plot(1:10,cumsum(100*PCTVAR(2,:)),'-bo');
        xlabel('Number of PLS components');
        ylabel('Percent Variance Explained in y');
        
        yfit = [ones(size(X,1),1) X]*beta;
        residuals = y - yfit;
        figure; stem(residuals)
        xlabel('Observation');
        ylabel('Residual');
        
        RSS = sum(residuals.^2);
        TSS = sum((y - mean(y)).^2);
        
    case 'CONN:cortical_pca'                  % Applies PCA to cortical activity profiles and create a surface map for each PC
        % the main purpose of this case is to have a better understanding
        % of the principal components
        % Example: sc1sc2_conn_model('CONN:cortical_pca');
        
        sn             = returnSubjs;
        glm            = 7;
        experiment_num = 1;
        xres           = 162;
        nPC            = 10;
        
        atlas_res = 32; % other option is 64?
        
        vararginoptions(varargin, {'sn', 'glm', 'experiment_num', 'xres', 'nPC', 'atlas_res'});
        
        experiment = sprintf('sc%d', experiment_num);
        
        % setting directories
        betaDir     = fullfile(baseDir, experiment, sprintf('Beta_GLM_%d', glm));
        surfDir     = fullfile(baseDir, experiment, sprintf('fs_LR_%d', atlas_res)); %% the directory where the atlas is located + giftti label files
        connDir     = fullfile(baseDir, experiment, 'connModels');
        %         encodingDir = fullfile(baseDir, experiment, encodeDir, sprintf('glm%d', glm));
        
        for s = sn
            fprintf('Cortical %d PCA loading map for %s\n', xres, subj_name{s});
            % directory to save the pca map for loadings
            connPcaSurfDir = fullfile(connDir, 'pca', sprintf('pc%d', nPC), subj_name{s}, 'surfMap');
            dircheck(connPcaSurfDir);
            % load in the beta file for the tesselation
            load(fullfile(betaDir, subj_name{s}, sprintf('beta_regions_tesselsWB%d', xres)));
            
            X = B.mbetasUW; % average beta for each tessel
            X = (X(:, 1:end - length(runLst)))'; % remove the intercepts
            
            % preparing X for PCA
            % do I need to standardize the data for each run separately?
            % if yes, then the following code needs to be modified
            X0 = bsxfun(@minus, X, nanmean(X));
            
            [coeff,score,latent,tsquared,explained,mu] = pca(X0, 'NumComponents',nPC);
            
            % create spatial maps for the loadings (coeff)
            
            % using workbench
            % lots of this part is like the ROI:define
            %%% saves Variance, VIF, and combined as a single gifti file which
            %%% is saved under directory connModels
            idx = 1;
            for h = 1
                
                G = gifti(fullfile(surfDir,sprintf('Icosahedron-%d.%dk.%s.label.gii', xres, atlas_res, hemI{h})));
                
                nReg = numel(unique(G.cdata))-1; % exclude 0 - medial wall
                
                dv = zeros(length(G.cdata), nPC);
                plottitle = cell(1, 10);
                
                for r = 1:nReg
                    
                    for ipc = 1:nPC
                        plottitle{ipc} = {sprintf('PC%d', ipc)};
                        dv(G.cdata(:,1) == r, ipc) = coeffL(idx, ipc);
                    end % ipc
                    idx = idx + 1;
                end % r(tessels)
                
                G       = surf_makeFuncGifti(dv,'anatomicalStruct', hemName{h}, 'columnNames', plottitle);
                outfile = (fullfile(connPcaSurfDir, sprintf('%s.cortPcLoadings-%d.pc%d.glm%d.%dk.%s.func.gii', ...
                    subj_name{s}, xres, nPC, glm, atlas_res, hemI{h})));
                %                 outfile = (fullfile(connPcaSurfDir, sprintf('%s.cortPcLoadings-32k.pc%d.glm%d.%dk.%s.func.gii', ...
                %                             subj_name{s}, nPC, glm, atlas_res, hemI{h})));
                save(G, outfile);
            end % h (hemispheres)
        end % sn
    case 'CONN:py:Cerebellar_pls'             % This case is used to transfer nifti images created in python to suit space
        % this case is temporary and I am writing it right now cause I
        % basically have not found any other way to efficiently translate
        % suit_map2surf code to python. So I create nifti images for the
        % loadings and in here, I will transfer them into flatmap
        % Example: sc1sc2_conn_model('CONN:py:Cerebellar_pls', 'xres', 162, 'npls', 7)
        xres = 162;
        npls = 7;
        
        vararginoptions(varargin, {'xres', 'npls'});
        
        mapDir   = '/Users/ladan/Documents/MATLAB/Projects/Connectivity/Connectivity_MDTB/python_code';
        giftiDir = fullfile(mapDir, 'preliminaryIMs');
        dircheck(giftiDir);
        
        for ic = 0:(npls -1)
            % load in the niftin image
            imageName = fullfile(giftiDir, sprintf('groupPLS%d_%d_%d.nii', npls, ic, xres));
            V         = spm_vol(imageName);
            
            % map it to flatmap
            Dsuit = suit_map2surf(V,'stats','nanmean');
            
            % make a gifti out of it
            wG = surf_makeFuncGifti(Dsuit, 'anatomicalStruct', 'Cerebellum');
            
            % save the gifti
            gName = fullfile(giftiDir, sprintf('group_PLS%d_%d_%d.func.gii', npls, ic, xres));
            save(wG, gName);
            
            columnName{ic+1}  = sprintf('PLS_comp_%d', ic);
            infilenames{ic+1} = gName;
        end
        % also make a group gifti
        outGroupName = fullfile(giftiDir, sprintf('all_group_PLS%d_%d.func.gii', npls, xres));
        surf_groupGiftis(infilenames, 'outfilenames', {outGroupName}, 'outcolnames', columnName, 'replaceNaNs', 1);
        
    otherwise
        disp('there is no such case.')
end
end

%==========================================================================
% Local functions
function dircheck(dir)
% used for creating directories!
if ~exist(dir,'dir')
    fprintf('%s doesn''t exist. Creating one now. \n',dir);
    mkdir(dir);
end
end % function dircheck

function [W, R2, R, R2_vox, R_vox, varargout] = conn_model_fit(Y,X,method,varargin)
% function [u,R2,R,R2_vox,R_vox,varargout]=sc1_connect_fit(Y,X,method,varargin)
%
% INPUT:
%    Y: NxP matrix Data
%    X: NxQ matrix for random effects
%    method: 'linRegress','winnerTakeAll','ridgeFixed', 'pcr', 'plsr_1', 'plsr_2'
% VARARGIN:
%    'threshold': Thresholds the u-coefficient at a particular value(s)
%                 before evaluating the prediction (all < threshold -> 0)
%    'numReg' : For nonNegStepwise the maximum number of regions
%    'lambda' : [L1 L2] regularization coefficient
% OUTPUT:
%    R2      : correlation value between Y-actual and Y-pred (overall)
%    R       : portion of correctly specified variance (overall)
%    R2_vox  : portion of correctly specified variance (voxels)
%    R_vox   : correlation values between Y-actual and Y-pred (voxels)
%    u       : regression coefficients
% Maedbh King (26/08/2016)
% joern.diedrichsen@googlemail.com
% Ladan Shahshahani (June 2020)


params = [];
scale  = 0; % 1 to zscore the data and 0 if not!

switch scale
    case 1 % if you want to scale the data
        Y = zscore(Y);
        X = zscore(X);
end

vararginoptions(varargin,{'params', 'scale'});

[N,M] = size(Y); % N # conditions, P # cerebellar voxels
[N,P] = size(X); % N # conditions, Q # cortical voxels/tessels

% P - M
% Q - P
% u - W

W        = zeros(P,M);
% features = [1:P];

% Estimate the weights
switch method
    case 'linRegress'              %  Normal linear regression
        W = (X'*X)\(X'*Y);
        
        varargout{1} = [];
    case 'ridgeFixed'              % L2 regression
        %             u = G*trainX'*((trainX*G*trainX'+eye(sum(trainIdx))*sigma2)\trainY);
        lambda = params(2);
        W = (X'*X + eye(P)*lambda)\(X'*Y);
        
        varargout{1} = [];
    case 'plsr_1'                  % pls regression with a diagonal B (one-to-one topography)
        % this case uses matlab's plsregress function
        
        [XL,YL,XS,YS,W,PCTVAR,MSE,stats] = plsregress(X,Y, params);
        
        % W (matlab's BETA) returned by plsregress contains the intercept
        % term in its first row. I'm discarding it here!
        W(1, :) = [];
        
        plsr_out.Xloading{1, 1} = XL;
        plsr_out.Yloading{1, 1} = YL;
        
        plsr_out.Xscore{1, 1} = XS;
        plsr_out.Yscore{1, 1} = YS;
        
        plsr_out.BETA{1, 1} = W;
        
        plsr_out.Xpctvar{1, 1} = PCTVAR(1, :);
        plsr_out.Ypctvar{1, 1} = PCTVAR(2, :);
        
        plsr_out.XMSE{1, 1} = MSE(1, :);
        plsr_out.YMSE{1, 1} = MSE(2, :);
        
        plsr_out.stats{1, 1} = stats;
        
        varargout{1} = plsr_out;
    case 'plsr_2'                  % "general" pls regression
    case 'pcr'                     % principal component regression
        nPC = params(1);
        
        % PCA to X
        [XL,XS,latent,tsquared,explained,mu] = pca(X, 'NumComponents',nPC);
        
        % Regress Y on scores of X
        W = (XS'*XS)\(XS'*Y);
        
        pcr_out.Xloading{1, 1}  = XL;
        pcr_out.latent{1, 1}    = latent;    % principal component variances
        pcr_out.tsquared{1, 1}  = tsquared;  % Hotelling's T-squared statistic for each observation in X
        pcr_out.explained{1, 1} = explained; % the percentage of the total variance explained by each principal component
        pcr_out.mu{1, 1}        = mu;        % estimated mean of each variable in X
        
        varargout{1} = pcr_out;
        X = XS;
        
    otherwise
        error ('unknown Method');
end

% Evaluate prediction by calculating R2 and R
SST = nansum(Y.*Y);

% for i = 1:size(W, 3) %% why 3??????
%     Ypred = X * W(:,:,i);
%     res   = Y - Ypred;
%     SSR   = nansum(res.^2);
%     R2_vox(i,:) = 1 - SSR./SST;
%     R2(i,1)     = 1 - nansum(SSR)/nansum(SST);
%
%     % R (per voxel)
%     SYP = nansum(Y.*Ypred,1);
%     SPP = nansum(Ypred.*Ypred);
%
%     R_vox(i,:) = SYP./sqrt(SST.*SPP);
%     R(i,1)     = nansum(SYP)./sqrt(nansum(SST).*nansum(SPP));
% end % i (size (W, 3))

Ypred = X * W;
res   = Y - Ypred;
SSR   = nansum(res.^2);

% R2
R2     = 1 - nansum(SSR)/nansum(SST);
R2_vox = 1 - SSR./SST; % per voxel

% R
SYP = nansum(Y.*Ypred,1);
SPP = nansum(Ypred.*Ypred);

R     = nansum(SYP)./sqrt(nansum(SST).*nansum(SPP));
R_vox = SYP./sqrt(SST.*SPP); % per voxel
end

function varargout = conn_model_reg(X, Y, varargin)
% function to implement different regression models
% INPUT:
%  Y                : Cortical Noise-normalized activation patternns (#conditions by #voxels/parcels)
%  X                : Cerebellar Noise-normalized activation patternns (#conditions by #voxels/parcels)
% for ols and rrr X include s
%  varargin:
%       model : 'ols' or 'rrr' or 'pls'
%       nPC   : # of PCs for rrr
%       lambda: for ridge regression is the lambda
%
% OUTPUT:
%   What              : connectivity weights

model  = 'ols'; % regression method
nPC    = 0;    % number of Pcs used in rrr
lambda = 0.01;  % lambda parameter in ridge regression

vararginoptions(varargin, {'model', 'nPC', 'lambda'});

if nPC == 0 % number of PCs hasn't been set
    nPC = min(size(X,1)-1,size(X,2));
end

switch model
    case 'ols'          % ols
        %         What = (X'*X)\(X'*Y);
        % in case X'X is singular, Matlab will raise a warning:
        % Warning: Matrix is close to singular or badly scaled. Results may be inaccurate.
        % I am using pinv for calculation.
        What = pinv(X'*X)*(X'*Y); % in case X'X is singular (not invertible)
        varargout{1} = What;
    case 'rrr'          % reduced rank regression
        % 1.1 Wols
        %         What_ols = (X'*X)\(X'*Y);
        What_ols = pinv(X'*X)*(X'*Y); % in case X'X is singular (not invertible)
        
        % 1.2 Yhat_OLS = XWhat_OLS
        Yhat_ols = X*What_ols;
        
        % 1.3 Apply PCA to Yhat_ols
        %[coef,score, latent] = pca(Yhat_ols);
        [coef,~, ~] = pca(Yhat_ols);
        
        % 1.4 retain the first r PCs (Ur)
        %Ur = score(:, 1:rPC);
        Ur = coef(:, 1:nPC);
        
        % 1.5 What_RRR = What_OLS (UrUr')
        What = What_ols * (Ur*Ur');
        varargout{1} = What;
    case 'ridge'        % ridge regression
        % inputs should be centered
        % pseudocode: What = (X'X + lambdaI)X'y
        What = pinv(X'*X + lambda*eye(size(X'*X)))*X'*Y;
        varargout{1} = What;
    case 'plsr_matlab'  % matlab's pls regression
        % inputs should be centered
        X0 = zscore(X);
        Y0 = zscore(Y);
        
        %         X0 = X;
        %         Y0 = Y;
        
        
        [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X0,Y0, nPC);
        
        plsr_out.Xloading = XL;
        plsr_out.Yloading = YL;
        
        plsr_out.Xscore = XS;
        plsr_out.Yscore = YS;
        
        plsr_out.BETA = BETA;
        
        plsr_out.Xpctvar = PCTVAR(1, :);
        plsr_out.Ypctvar = PCTVAR(2, :);
        
        plsr_out.XMSE = MSE(1, :);
        plsr_out.YMSE = MSE(2, :);
        
        plsr_out.stats = stats;
        
        varargout{1} = plsr_out;
    case 'plsr'         % my pls regression using SVD
        % mostly based on (Krishnan et al., 2011) paper
        % z-score X and Y
        X0 = zscore(X);
        Y0 = zscore(Y);
        
        iterate = true;
        Xi = X0;
        Yi = Y0;
        
        i = 1;
        L = rank(X0);
        
        T = []; % X scores
        U = []; % Y scores
        P = []; % X loadings
        W = []; % X weights
        C = []; % Y weights
        b = []; % regression coefficients
        
        while iterate
            % inspecting just one iteration
            %%% matrix of covariance
            Ri = Xi'*Yi;
            
            %%% SVD of covariance matrix
            [Wi, D, Ci] = svd(Ri);
            
            %%% the first X score
            wi = Wi(:, 1);
            ci = Ci(:, 1);
            ti = (Xi*wi);
            ti = ti/sqrt(ti'*ti);
            
            %%% The loading of X0 on ti
            pi = Xi'*ti;
            
            %%% ith latent variable of Y
            ui = Yi*ci;
            
            %%% reconstruct X and Y
            Yihat = ui*ci';
            Xihat = ti*pi';
            
            %%% estimate bi
            bi = ti'*ui;
            
            % update T, U, P, W, C, b
            T = [T, ti];
            U = [U, ui];
            P = [P, pi];
            W = [W, wi];
            C = [C, ci];
            b = [b, bi];
            
            i = i + 1;
            
            % deflation
            Xi = Xi - Xihat;
            %                     Yi = Yi - Yihat;
            
            if size(W, 2) == L
                iterate = false;
            end
        end % iterate
        
        % B: the diagonal matrix
        B = diag(b);
        
        Yhat = T*B*C';
        Wpls = pinv(P')*B*C';
        
        plsr_out.Yhat     = Yhat;
        plsr_out.Xscores  = T;
        plsr_out.Yscores  = U;
        plsr_out.Xloading = P;
        plsr_out.Yloading = C;
        plsr_out.B        = B;
        plsr_out.Wpls     = Wpls;
        
        varargout{1} = plsr_out;
        
end % regression model
end % implement regression model

function [R2cv, Rcv, R2, R, What] = conn_model_cv(X, Y, folds, condition, varargin)
% INPUT:
%  Y                : Cortical Noise-normalized activation patternns (#conditions by #voxels/parcels)
%  X                : Cerebellar Noise-normalized activation patternns (#conditions by #voxels/parcels)
%  folds            : vector including the partitions/folds (run numbers. Use SPM_info).
%  condition        : vector of conditions (Use SPM_info)
%  varargin:
%       model  : 'ols' or 'rrr' or 'ridge' or 'PLS'
%       nPC    : # of PCs for rrr
%       lambda : lambda for ridge regression
%
% OUTPUT:
%   R2cv              : cross validated R2
%   R                 :
% does the cross validation

removeMean = 1;     % remove mean from activity
model      = 'ols';
nPC        = 0;
lambda     = 0.01;

vararginoptions(varargin, {'removeMean', 'model', 'nPC', 'lambda'});

part    = unique(folds)';
part    = part(part~=0)';     % runs with index 0 are discarded.
numPart = numel(part);        % number of folds (runs)
cond    = unique(condition)';
cond    = cond(cond~=0)';     % add 1 to the T.cond if you don't want to exclude the instructions
numCond = numel(cond);

[~,numVoxx]   = size(X);
[~,numVoxy]   = size(Y);

Ax = zeros(numCond,numVoxx,numPart);
Ay = zeros(numCond,numVoxy,numPart);
Z  = indicatorMatrix('identity_p',condition);

% Estimate condition means within each run
for i=1:numPart
    Za = Z(folds==part(i),:);
    Xa = X(folds==part(i),:);
    Ya = Y(folds==part(i),:);
    
    Ax(:,:,i) = pinv(Za)*Xa;
    Ay(:,:,i) = pinv(Za)*Ya;
    if (removeMean) % per partition
        Ax(:,:,i)=bsxfun(@minus,Ax(:,:,i),sum(Ax(:,:,i),1)/numCond);
        Ay(:,:,i)=bsxfun(@minus,Ay(:,:,i),sum(Ay(:,:,i),1)/numCond);
    end
end

Yhat_cv = zeros(numCond,numVoxy,numPart);
What_cv = zeros(numVoxx,numVoxy,numPart);
for ifold = 1:numPart
    % run indices used for training
    trainRun = part ~=ifold;
    testRun  = part == ifold;
    
    % get the runs data for training
    Xtrains = Ax(:, :, trainRun);
    Ytrains = Ay(:, :, trainRun);
    
    % get the runs data for testing
    Xtest = Ax(:, :, testRun);
    Ytest = Ay(:, :, testRun);
    
    % calculate the mean across training runs
    Xtrain = nanmean(Xtrains, 3);
    Ytrain = nanmean(Ytrains, 3);
    
    % modeling
    switch model
        case 'ols'          % ols regression
            What_cv(:, :, ifold) = conn_model_reg(Xtrain, Ytrain, 'model', 'ols');
            Yhat_cv(:, :, ifold) = Xtest*What_cv(:, :, ifold);
        case 'rrr'          % reduced rank regression
            What_cv(:, :, ifold) = conn_model_reg(Xtrain, Ytrain, 'model', 'rrr', 'nPC', nPC);
            Yhat_cv(:, :, ifold) = Xtest*What_cv(:, :, ifold);
        case 'ridge'        % ridge regression
            What_cv(:, :, ifold) = conn_model_reg(Xtrain, Ytrain, 'model', 'ridge', 'lambda', lambda);
            Yhat_cv(:, :, ifold) = Xtest*What_cv(:, :, ifold);
        case 'plsr_matlab'  % matlab's pls regression
            out_cv{ifold} = conn_model_reg(Xtrain, Ytrain, 'model', 'plsr_matlab', 'nPC', nPC);
            What_cv(:, :, ifold) = out_cv{ifold}.BETA;
            Yhat_cv(:, :, ifold) = Ytest - out_cv{ifold}.stats.Yresiduals;
        case 'plsr'         % pls regression based on (Krishnan et al., 2011)
    end % switch model
    
end % ifold

What = nanmean(What_cv, 3);

% calculate R2cv and Rcv
SSR      = sum(sum(sum((Ay-Yhat_cv).^2)));
SST      = sum(sum(sum(Ay.^2)));
SSP      = sum(sum(sum(Yhat_cv.^2)));
SSPA     = sum(sum(sum(Yhat_cv.*Ay)));

R2cv = 1 - SSR/SST;
Rcv  = SSPA / sqrt(SST*SSP);

% calculate R2 and R
%%% fit the model using the whole data
Axm = nanmean(Ax, 3);
Aym = nanmean(Ay, 3);
%%% modeling
switch model
    case 'ols'          % ols regression
        What = conn_model_reg(Axm, Aym, 'model', 'ols');
        %%% get the predicted Y
        Yhat = Axm*What;
    case 'rrr'          % reduced rank regression
        What = conn_model_reg(Axm, Aym, 'model', 'rrr', 'nPC', nPC);
        %%% get the predicted Y
        Yhat = Axm*What;
    case 'ridge'        % ridge regression
        What = conn_model_reg(Axm, Aym, 'model', 'ridge', 'lambda', lambda);
        %%% get the predicted Y
        Yhat = Axm*What;
    case 'plsr_matlab'  % partial least squares regression using MATLAB's built-in function
        out  = conn_model_reg(Xtrain, Ytrain, 'model', 'plsr_matlab', 'nPC', nPC);
        What = out.BETA;
        Yhat = Ytest - out.stats.Yresiduals;
end % switch model

% calculate R2cv and Rcv
SSR      = sum(sum(sum((Aym-Yhat).^2)));
SST      = sum(sum(sum(Aym.^2)));
SSP      = sum(sum(sum(Yhat.^2)));
SSPA     = sum(sum(sum(Yhat.*Aym)));

R2 = 1 - SSR/SST;
R  = SSPA / sqrt(SST*SSP);
end % doing cross validation

%==========================================================================
%==========================================================================
% Instructions
% 1. Use 'Houskeeping:renameSPM' to change the directories in the SPM.mat
% 2. Use 'ROI:mdtb:define' to create Region files for tesselsWB with
% different numbers of nodes. These will be used in estimating connectivity
% weights
% 3. Use 'ROI:mdtb:beta_unn' to get the univariately prewhitened betas for
% each region: cerebellum_grey and cortical tessels: tesselsWB162,
% tesselsWB 362, ...
% 4. Make sure that you have changed suit_reslice_dartel.m according to the
% instructions below.
% 5. Using PREP cases, normalize the beta values for cerebellum_grey into
% suit space.
% 6. Create Y_info, containing beta values for both the cerebellum and
% cortex (Using PREP cases).
% 7. Extract mean beta for cross-session modeling. calculate mean beta for
% each session (averaging across runs within a session).
% 8. use CONN:MDTB:model_mbeta to generate different connectivity models.
% Two modes of training can be used: crossed: Xtrain from one session
% paired with Ytrain from the other session. See how the integrated
% structure for conditions information is created!
% **** can use 'CONN:MDTB:run_model_mbeta' to generate different models
% 9. Evaluate!

%==========================================================================
%==========================================================================
% Instructions to change toolboxes functions (if needed)
%==========================================================================
% To run 'PREP:mdtb:cereb:suit_betas' which creates 'wdBetas_UW.mat', you
% need to make a change in suit_reslice_dartel.m function. You will need to
% add a variable as output. Compare the following code with the original
% suit function, the bounding box part:
% Now get the output images into the right resolution and bounding box
% for i=1:N
%     V=spm_vol(out.files{i});
%     Data=zeros(size(Xm));
%     for z=1:size(Xm,3);
%         Data(:,:,z)=spm_sample_vol(V,Xm(:,:,z),Ym(:,:,z),Zm(:,:,z),job.interp);
%     end;
%     V.fname=finalname{i};
%     V.dim=size(Xm);
%     % Determine the transformation matrix from 4 points
%     % This is safe, but can probably be done more elegantly
%     i1=V.dim(1);
%     i2=V.dim(2);
%     i3=V.dim(3);
%     x=[[X(1,1,1);Y(1,1,1);Z(1,1,1);1],...
%         [X(i1,1,1);Y(i1,1,1);Z(i1,1,1);1],...
%         [X(i1,i2,1);Y(i1,i2,1);Z(i1,i2,1);1],...
%         [X(i1,i2,i3);Y(i1,i2,i3);Z(i1,i2,i3);1]];
%     v=[[1;1;1;1],...
%         [i1;1;1;1],...
%         [i1;i2;1;1],...
%         [i1;i2;i3;1]];
%     V.mat=x*pinv(v);
%     D(i, :, :, :) = Data;
%     spm_write_vol(V,Data);
%     delete(out.files{i});
%     delete(J.images{i}{1});
% end;
% varargout{1} = D;
%==========================================================================
% making necessary changes to spm_defaults.m function:
% first open spm_defaults by typing >> open spm_defaults.m
% on line 49 (alternatively, you can search for defaults.mat.format), set
% the format to '-v7.3'. Otherwise, the big mat files will be saved as
% empty!
%==========================================================================
% creating the new text file with the information about the
% conditions/tasks:
% The new text file includes new flags: inst flag indicates whether the
% condition is a instruction or not. instOrder specifies the order of the
% task in the subject.
% The other fields from sc1_sc2_taskConds_GLM7 were also changed. in
% taskNum, condNum, taskNumUni, and condNumUni, instruction is always
% indicated with 0 (just as before).
% For the overlap flag, the instructions that is comming before a shared
% task also has 1 as overlap. Because, technically I think it can also be
% considered as a shared task/condition.
% This code is using sc1_sc2_taskConds_GLM7 in the evaluation cases. For
% future use, sc1_sc2_taskConds_conn.txt will be used!
%==========================================================================
% 'ROI:MDTB:add_to_beta' uses the beta_region files already created to
% create a new dataa structure, like Y_info files saved in encoding
% directory. These files can be used in codes written in python! The files
% will be saved as 'Y_info_glm#_roi.mat'

