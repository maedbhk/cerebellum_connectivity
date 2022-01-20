function [ varargout ] = sc1sc2_makeflatmaps( what, varargin )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%==========================================================================
% setting directories
% baseDir         = '/Volumes/MotorControl/data/super_cerebellum_new';
% baseDir         = '/Users/ladan/Documents/Project-Cerebellum/Cerebellum_Data';
baseDir         = '/Volumes/diedrichsen_data$/data/super_cerebellum/'; 
atlasDir        = '/srv/diedrichsen/data/Atlas_templates/fs_LR_32';
catlasDir        = '/Users/jdiedrichsen/Data/cerebellar_atlases';

suitDir         = '/Users/jdiedrichsen/Matlab/imaging/suit/flatmap';

switch (what) 
    case 'MDTB_regions'
        region = varargin{1}; 
        fname = fullfile(catlasDir,'King_2019','atl-MDTB10_dseg.label.gii'); 
        fname = fullfile(suitDir,'MDTB_10Regions.label.gii'); 
        G=gifti(fname); 
        data = G.cdata; 
        data(data~=region)=NaN; 
        set(gcf,'PaperPosition',[2 2 6.8 5.8]);
        set(gca,'Color',[0 0 0])
        suit_plotflatmap(data,'type','label','cmap',G.labels.rgba(2:end,:),...
            'borderstyle','w.','ylims',[-85 85],'xlims',[-100 100],'underscale',[-0.8 2.3]); 
        wysiwyg; 
end; 