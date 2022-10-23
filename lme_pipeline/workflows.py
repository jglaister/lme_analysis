
import os  # system functions
import shutil
from glob import glob

import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl
import nipype.interfaces.ants as ants
import nipype.interfaces.utility as util

#from nipype import Workflow, Node, MapNode, Function, IdentityInterface
#from nipype.interfaces.ants import N4BiasFieldCorrection, Registration, ApplyTransforms
#from nipype.interfaces.fsl import Smooth, Split
#from nipype.interfaces.ants import MultiplyImages
#from nipype.interfaces.utility import Split as SplitList

from lme_pipeline.interfaces import ProcessMTR, ConvertTransformFile, ProcessLME

class PipelineWorkflow(pe.Workflow):
    def __init__(self, name, scan_directory, patient_id=None, scan_id=None):
        self.scan_directory = scan_directory
        self.patient_id = patient_id if patient_id is not None else ''
        if scan_id is None or scan_id == '':
            self.scan_id = ''
        else:
            self.scan_id = scan_id
            name += '_' + scan_id
        base_dir = os.path.join(scan_directory, self.patient_id, 'pipeline')
        super(PipelineWorkflow, self).__init__(name, base_dir)
        self.config['execution']['crashdump_dir'] = os.path.join(self.base_dir, self.name)

    def clean(self):
        shutil.rmtree(os.path.join(self.base_dir, self.name))
        if os.path.basename(self.base_dir) == 'pipeline' and os.listdir(self.base_dir) == []:
            shutil.rmtree(self.base_dir)


def create_lme_metrics_workflow(scan_directory, patient_id, scan_id, image_flag=False, coord=None):#, reg_mt=True):
    """


    Python implementation of the CVS segmentation from Taki's group
segment_cvs -d /Users/jiwonoh/Documents/CVS_test -t1 /Users/jiwonoh/Desktop/Test_CAVSMS_dicom/18101CAVMS010_18101CAVMS010/NIFTI/RESEARCH_RESEARCH_NEURO_3D_T1_MPRAGE_20190808121525_5.nii.gz  -epi /Users/jiwonoh/Desktop/Test_CAVSMS_dicom/18101CAVMS010_18101CAVMS010/NIFTI/RESEARCH_RESEARCH_NEURO_3D_T2STAR_segEPI_20190808121525_7.nii.gz  -flair /Users/jiwonoh/Desktop/Test_CAVSMS_dicom/18101CAVMS010_18101CAVMS010/NIFTI/RESEARCH_RESEARCH_NEURO_3D_T2_FLAIR_20190808121525_6.nii.gz

/Users/jiwonoh/Desktop/Test_CAVSMS_dicom/18101CAVMS010_18101CAVMS010/RESEARCH_RESEARCH_NEURO_3D_T1_MPRAGE_20190808121525_5.nii.gz
"""



    name = 'lme_metrics' + scan_id
    #If patient id and scan id are provided, assume an IACL folder structure
    #Otherwise assume that we are saving in the specified directory
    if patient_id is not None and scan_id is not None:
        #base_dir = os.path.join(scan_directory, patient_id, 'pipeline')
        #name += '_' + scan_id
        scan_path = os.path.join(scan_directory, patient_id, scan_id)
    else:
        scan_path = scan_directory

    wf = PipelineWorkflow(name, scan_directory, patient_id, scan_id)
    #wf = Workflow(name, scan_directory)

    input_node = pe.Node(util.IdentityInterface(fields=['image_files'], mandatory_inputs=False), 'input_node')
    #compute_coord_flag = True if coord is not None else False
    #compute_t2star_flag = True if input_node.inputs.t2star_image is not None else False
    #compute_mtr_flag = True if input_node.inputs.mton_image is not None else False
    #split_mt_flag = True if compute_mtr_flag and input_node.inputs.mtoff_image is None else False

    # Grab needed results from IACL folder
    t1_image = glob(os.path.join(scan_path, '*_MPRAGEPre_reg.nii.gz'))[0]
    bm_image = glob(os.path.join(scan_path, '*_MPRAGEPre_reg_mask.nii.gz'))[0]
    flairpg_image = glob(os.path.join(scan_path, '*_FLAIRPost_3D_reg.nii.gz'))[0]
    flairpg_mat = glob(os.path.join(scan_path, '*_FLAIRPost_3D_reg.mat'))[0]
    flair_tf = glob(os.path.join(scan_path, '*_FLAIRPost_3D_reg.mat'))[0]

    # Get transformation from FLAIR space to T1 space
    convert_tf = pe.Node(ConvertTransformFile(), 'convert_tf')
    convert_tf.inputs.input_transform_file = flair_tf
    convert_tf.inputs.hm = True
    convert_tf.inputs.ras = True
    wf.add_nodes([convert_tf])

    process_lme = pe.Node(ProcessLME(), 'process_lme')
    process_lme.inputs.scan_path = scan_path
    process_lme.inputs.coordinate = coord
    wf.connect(convert_tf, 'transform_file', process_lme, 'transform_file')
    if image_flag:
        wf.connect(input_node, 'image_files', process_lme, 'image_files')

    # Output node to save outputs
    output_node = pe.Node(util.IdentityInterface(fields=['transform_file', 'lme_csv']),
                          mandatory_inputs=False, name='output_node')

    wf.connect(convert_tf, 'transform_file', output_node, 'transform_file')

    # bias_correction_t1 = Node(N4BiasFieldCorrection(), "bias_t1")
    # bias_correction_t1.inputs.shrink_factor = 4
    # bias_correction_t1.inputs.n_iterations = [200, 200, 200, 200]
    # bias_correction_t1.inputs.convergence_threshold = 0.0005
    # wf.connect(inputnode, 't1_image', bias_correction_t1, 'input_image')
    #
    # bias_correction_epi = Node(N4BiasFieldCorrection(), "bias_epi")
    # bias_correction_epi.inputs.shrink_factor = 4
    # bias_correction_epi.inputs.n_iterations = [200, 200, 200, 200]
    # bias_correction_epi.inputs.convergence_threshold = 0.0005
    # wf.connect(inputnode, 'epi_image', bias_correction_epi, 'input_image')
    #
    # bias_correction_flair = Node(N4BiasFieldCorrection(), "bias_flair")
    # bias_correction_flair.inputs.shrink_factor = 4
    # bias_correction_flair.inputs.n_iterations = [200, 200, 200, 200]
    # bias_correction_flair.inputs.convergence_threshold = 0.0005
    # wf.connect(inputnode, 'flair_image', bias_correction_flair, 'input_image')

    #TODO: Split registration to new workflow

    # register_flag = False
    # if register_flag:
    #     #MTR split file
    #     split_mt = Node(Split(), 'split_mt')
    #     split_mt.inputs.dimension = 't'
    #     wf.connect([(input_node, split_mt, [('mtr_image', 'in_file')])])
    #
    #     split_mt_list = Node(SplitList(), 'split_mt_list')
    #     split_mt_list.inputs.splits = [1, 1]
    #     split_mt_list.inputs.squeeze = True
    #     wf.connect([(split_mt, split_mt_list, [('out_files', 'inlist')])])
    #
    #     #Register flair and EPI to T1
    #     #Affine instead of rigid
    #     reg_mtoff_to_t1 = Node(Registration(), "mtoff_reg")
    #     reg_mtoff_to_t1.inputs.initial_moving_transform_com = 1
    #     reg_mtoff_to_t1.inputs.metric = ['MI', 'MI']
    #     reg_mtoff_to_t1.inputs.metric_weight = [1.0, 1.0]
    #     reg_mtoff_to_t1.inputs.radius_or_number_of_bins = [32, 32]
    #     reg_mtoff_to_t1.inputs.sampling_strategy = ['Regular', 'Regular']
    #     reg_mtoff_to_t1.inputs.sampling_percentage = [0.1, 0.1]
    #     reg_mtoff_to_t1.inputs.transforms = ['Rigid', 'Affine']
    #     reg_mtoff_to_t1.inputs.transform_parameters = [(0.1,), (0.1,)]
    #     reg_mtoff_to_t1.inputs.number_of_iterations = [[100, 75, 50, 25],
    #                                                     [100, 75, 50, 25]]
    #     reg_mtoff_to_t1.inputs.convergence_threshold = [1.e-6, 1.e-6]
    #     reg_mtoff_to_t1.inputs.convergence_window_size = [10, 10]
    #     reg_mtoff_to_t1.inputs.smoothing_sigmas = [[3, 2, 1, 0], [3, 2, 1, 0]]
    #     reg_mtoff_to_t1.inputs.sigma_units = ['vox', 'vox']
    #     reg_mtoff_to_t1.inputs.shrink_factors = [[8, 4, 2, 1], [8, 4, 2, 1]]
    #     reg_mtoff_to_t1.inputs.winsorize_upper_quantile = 0.99
    #     reg_mtoff_to_t1.inputs.winsorize_lower_quantile = 0.01
    #     reg_mtoff_to_t1.inputs.collapse_output_transforms = True  # For explicit completeness
    #     reg_mtoff_to_t1.inputs.float = True
    #     reg_mtoff_to_t1.inputs.use_estimate_learning_rate_once = [True, True]
    #     reg_mtoff_to_t1.inputs.output_warped_image = True
    #     reg_mtoff_to_t1.inputs.fixed_image = t1_image
    #     #TODO: If mtr
    #     wf.connect([(split_mt_list, reg_mtoff_to_t1, [('out1', 'moving_image')])])
    #
    #     transform_mton = Node(ApplyTransforms(),'transform_mton')
    #     #transform_mton.inputs.invert_transform_flags = True
    #     transform_mton.inputs.interpolation = 'Linear'
    #     transform_mton.inputs.default_value = 0
    #     transform_mton.inputs.reference_image = t1_image
    #     wf.connect([(reg_mtoff_to_t1, transform_mton, [('forward_transforms', 'transforms')])])
    #     wf.connect([(split_mt_list, transform_mton, [('out2', 'input_image')])])
    #
    #     #Register flair and EPI to T1
    #     #Affine instead of rigid
    #     reg_t2star_to_t1 = Node(Registration(), "t2star_reg")
    #     reg_t2star_to_t1.inputs.initial_moving_transform_com = 1
    #     reg_t2star_to_t1.inputs.metric = ['MI', 'MI']
    #     reg_t2star_to_t1.inputs.metric_weight = [1.0, 1.0]
    #     reg_t2star_to_t1.inputs.radius_or_number_of_bins = [32, 32]
    #     reg_t2star_to_t1.inputs.sampling_strategy = ['Regular', 'Regular']
    #     reg_t2star_to_t1.inputs.sampling_percentage = [0.1, 0.1]
    #     reg_t2star_to_t1.inputs.transforms = ['Rigid', 'Affine']
    #     reg_t2star_to_t1.inputs.transform_parameters = [(0.1,), (0.1,)]
    #     reg_t2star_to_t1.inputs.number_of_iterations = [[100, 75, 50, 25],
    #                                                    [100, 75, 50, 25]]
    #     reg_t2star_to_t1.inputs.convergence_threshold = [1.e-6, 1.e-6]
    #     reg_t2star_to_t1.inputs.convergence_window_size = [10, 10]
    #     reg_t2star_to_t1.inputs.smoothing_sigmas = [[3, 2, 1, 0], [3, 2, 1, 0]]
    #     reg_t2star_to_t1.inputs.sigma_units = ['vox', 'vox']
    #     reg_t2star_to_t1.inputs.shrink_factors = [[8, 4, 2, 1], [8, 4, 2, 1]]
    #     reg_t2star_to_t1.inputs.winsorize_upper_quantile = 0.99
    #     reg_t2star_to_t1.inputs.winsorize_lower_quantile = 0.01
    #     reg_t2star_to_t1.inputs.collapse_output_transforms = True  # For explicit completeness
    #     reg_t2star_to_t1.inputs.float = True
    #     reg_t2star_to_t1.inputs.use_estimate_learning_rate_once = [True, True]
    #     reg_t2star_to_t1.inputs.output_warped_image = True
    #     reg_t2star_to_t1.inputs.fixed_image = t1_image
    #     wf.connect([(input_node, reg_t2star_to_t1, [('t2star_image', 'moving_image')])])
    #
    #     #MTR interface
    #     process_mtr = Node(ProcessMTR(), "process_mtr")
    #     process_mtr.inputs.bm_file = bm_image
    #     wf.connect([(transform_mton, process_mtr, [('output_image', 'mton_file')])])
    #     wf.connect([(reg_mtoff_to_t1, process_mtr, [('warped_image', 'mtoff_file')])])
    #


    return wf




