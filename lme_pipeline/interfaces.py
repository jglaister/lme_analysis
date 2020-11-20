import os
import vtk
import nibabel as nib
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from sklearn.linear_model import LogisticRegression
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import csv
from glob import glob

from nipype.interfaces.base import (TraitedSpec, File, traits, isdefined, BaseInterface,
                                    BaseInterfaceInputSpec, CommandLineInputSpec, Directory)
from nipype.interfaces.elastix.utils import EditTransformInputSpec, EditTransformOutputSpec
from nipype.utils.filemanip import split_filename
from nipype.interfaces.ants.registration import ANTSCommand, ANTSCommandInputSpec

#from .base import ImageMathInputSpec, ImageMath, JISTCommand, MCRCommand
#TODO MTR interface


class ProcessMTRInputSpec(BaseInterfaceInputSpec):
    mton_file = File(exists=True, desc='file to threshold', mandatory=True)
    mtoff_file = File(exists=True, desc='file to threshold', mandatory=True)
    bm_file = File(exists=True, desc='file to threshold', mandatory=True)


class ProcessMTROutputSpec(TraitedSpec):
    output_mtr = File(exists=True, desc='binary mask')


class ProcessMTR(BaseInterface):
    input_spec = ProcessMTRInputSpec
    output_spec = ProcessMTROutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib

        mtoff_obj = nib.load(self.inputs.mtoff_file)
        mtoff_data = mtoff_obj.get_fdata()#.astype(np.float32)
        mton_data = nib.load(self.inputs.mton_file).get_fdata()#.astype(np.float32)
        bm_data = nib.load(self.inputs.bm_file).get_fdata()#.astype(np.float32)
        mtr = (mtoff_data - mton_data) / mtoff_data
        mtr = mtr * bm_data
        mtr[mtr<0] = 0

        out_name = split_filename(self.inputs.mtoff_file)[1] + '_mtr.nii.gz'
        nib.Nifti1Image(mtr, mtoff_obj.affine, mtoff_obj.header).to_filename(out_name)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_mtr'] = os.path.abspath(split_filename(self.inputs.mtoff_file)[1] + '_mtr.nii.gz')

        return outputs

class ConvertTransformFileInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, argstr="%d", usedefault=True, position=1, desc="image dimension (2 or 3)")
    input_transform_file = File(exists=True, mandatory=True, argstr="%s", position=2, desc="Fixed image or source image or reference image")
    output_transform_file = File(genfile=True, argstr="%s", position=3, desc="Fixed image or source image or reference image")
    hm = traits.Bool(False, usedefault=True, argstr='--hm')
    ras = traits.Bool(False, usedefault=True, argstr='--RAS')
    # HM and RAS options

class ConvertTransformFileOutputSpec(TraitedSpec):
    transform_file = File(desc="Compound transformation file")
    #ConvertTransformFile 3 /Volumes/LACIESHARE/RISStudy/RIS02047/01/RIS02047_01_STIR_reg.mat /Volumes/LACIESHARE/RISStudy/RIS02047/01/RIS02047_01_STIR_reg.txt --hm --RAS

class ConvertTransformFile(ANTSCommand):
    _cmd = "ConvertTransformFile"
    input_spec = ConvertTransformFileInputSpec
    output_spec = ConvertTransformFileOutputSpec

    def _gen_filename(self, name):
        if name == "output_transform_file":
            _, name, ext = split_filename(os.path.abspath(self.inputs.input_transform_file))
            return "".join((name, '.txt'))
        return None

    #def _format_arg(self, name, spec, value):
        # TODO Use genfile instead?
        #print(name)
        #if name == "output_transform_file" and self.inputs.output_transform_file == '':
        #    value = split_filename(self.inputs.input_transform_file)[1] + '.txt'
        #return super(ConvertTransformFile, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.output_transform_file):
            outputs["transform_file"] = os.path.abspath(self.inputs.output_transform_file)
            print('G')
        else:
            outputs["transform_file"] = os.path.abspath(self._gen_filename("output_transform_file"))
            print(outputs["transform_file"])
            print('H')

        return outputs
        #outputs['transform_file'] = self.inputs.output_transform_file


class ProcessLMEInputSpec(BaseInterfaceInputSpec):
    mtr_file = File(exists=True, desc='file to threshold', mandatory=True)
    t2star_file = File(exists=True, desc='file to threshold', mandatory=True)
    scan_path = Directory(exists=True, desc='file to threshold', mandatory=True)#traits.Str('combined.mat', desc='output file name', argstr='%s', usedefault=True)
    transform_file = File(exists=True, desc='file to threshold', mandatory=True)
    prefix = traits.String('output', usedefault=True)
    coordinate = traits.Tuple(traits.Int, traits.Int, traits.Int, mandatory=True)

class ProcessLMEOutputSpec(TraitedSpec):
    lme_csv = File(exists=True, desc='binary mask')

class ProcessLME(BaseInterface):
    input_spec = ProcessLMEInputSpec
    output_spec = ProcessLMEOutputSpec

    def _run_interface(self, runtime):

        # Get necessary files
        pgflair_orig_file = glob(os.path.join(self.inputs.scan_path, 'raw', '*_FLAIRPost_3D.nii.gz'))[0]

        macruise_file = glob(os.path.join(self.inputs.scan_path, '*_MPRAGEPre_reg_macruise.nii.gz'))[0]
        pgflair_file = glob(os.path.join(self.inputs.scan_path, '*_FLAIRPost_3D_reg.nii.gz'))[0]

        # Levelset files
        outer_ls_file = glob(os.path.join(self.inputs.scan_path, '*_MPRAGEPre_reg_macruise_outer.nii.gz'))[0]
        central_ls_file = glob(os.path.join(self.inputs.scan_path, '*_MPRAGEPre_reg_macruise_central.nii.gz'))[0]
        inner_ls_file = glob(os.path.join(self.inputs.scan_path, '*_MPRAGEPre_reg_macruise_inner.nii.gz'))[0]

        outer_file = glob(os.path.join(self.inputs.scan_path, '*_MPRAGEPre_reg_macruise_outer.vtk'))[0]
        central_file = glob(os.path.join(self.inputs.scan_path, '*_MPRAGEPre_reg_macruise_central.vtk'))[0]
        inner_file = glob(os.path.join(self.inputs.scan_path, '*_MPRAGEPre_reg_macruise_inner.vtk'))[0]

        coord_orig = self.inputs.coordinate

        transform = []
        with open(self.inputs.transform_file) as textFile:
            for line in textFile:
                transform.append([float(i) for i in line.split()])

        pgflair_orig_obj = nib.load(pgflair_orig_file)
        pgflair_orig_data = pgflair_orig_obj.get_fdata()
        Qform_orig = pgflair_orig_obj.affine

        pgflair_obj = nib.load(pgflair_file)
        pgflair_data = pgflair_obj.get_fdata()
        Qform = pgflair_obj.affine

        # Voxel location in MNI atlas space
        p_orig = np.matmul(Qform_orig,coord_orig + (1,))
        p_new = np.matmul(np.linalg.inv(transform),p_orig)
        coord = np.round(np.matmul(np.linalg.inv(Qform),p_new)[0:3]).astype(int)

        # TODO: Separate MSP to its own interface
        macruise_obj = nib.load(macruise_file)
        macruise_data = macruise_obj.get_fdata()
        wm_X = np.vstack((np.argwhere(macruise_data == 44), np.argwhere(macruise_data == 45)))
        wm_y = np.concatenate((np.ones(np.sum(macruise_data == 44)),np.zeros(np.sum(macruise_data == 45))))

        # Estimate MSP as a vertical line using logistic regression on the left and right WM coordinates
        # TODO: Allow for rotation?
        log_reg = LogisticRegression()
        log_reg.fit(wm_X[:, 0].reshape(-1, 1), wm_y)

        parameters = log_reg.coef_[0]
        parameter0 = log_reg.intercept_
        msp = -parameter0/parameters

        opp = np.empty_like(coord)
        opp[:] = coord
        opp[0] = np.round(2*msp - coord[0])


        #Load VTK files
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(outer_file)
        reader.Update()
        reader.GetOutput()
        nodes_vtk_array = reader.GetOutput().GetPoints().GetData()
        nodes_numpy_array = vtk_to_numpy(nodes_vtk_array)

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(central_file)
        reader.Update()
        reader.GetOutput()
        nodes_vtk_array = reader.GetOutput().GetPoints().GetData()
        nodes_numpy_array_cen = vtk_to_numpy(nodes_vtk_array)

        #Determine central surface coordinates
        if coord[0] > msp: #Limit central surface coordinates to only those on the same side as coord
            outer_nodes = nodes_numpy_array[nodes_numpy_array[:,0] > msp, :]
        else:
            outer_nodes = nodes_numpy_array[nodes_numpy_array[:,0] < msp, :]

        n = np.argmin(np.sum((outer_nodes - coord)**2, 1))
        coord_out = outer_nodes[n, :]
        n = np.argmin(np.sum((nodes_numpy_array_cen - coord_out)**2, 1))
        coord_cen = nodes_numpy_array_cen[n, :]
        coord_cen_round = np.round(coord_cen).astype(int)
        coord_class = macruise_data[coord_cen_round[0], coord_cen_round[1], coord_cen_round[2]]
        print('Coord class: ' + str(coord_class))

        #This code block forces the opposite class to have the same class as coord
        #if coord_class%2==0:
        #    opp_class = coord_class + 1
        #else:
        #    opp_class = coord_class - 1
        #nodes_numpy_array_round = np.round(nodes_numpy_array_cen).astype(int)
        #central_class = macruise_data[nodes_numpy_array_round[:,0],nodes_numpy_array_round[:,1],nodes_numpy_array_round[:,2]]
        #nodes_numpy_array_cen_c = nodes_numpy_array_cen[central_class == opp_class,:]
        #n =  np.argmin(np.sum((nodes_numpy_array_cen_c - opp)**2, 1))
        #opp_cen = nodes_numpy_array_cen_c[n, :]

        if coord[0]>msp: #Opposite of above to ensure that opp is across the MSP
            outer_nodes = nodes_numpy_array[nodes_numpy_array[:,0] < msp, :]
        else:
            outer_nodes = nodes_numpy_array[nodes_numpy_array[:,0] > msp, :]
        n = np.argmin(np.sum((outer_nodes - opp)**2, 1))
        opp_out = outer_nodes[n, :]
        n = np.argmin(np.sum((nodes_numpy_array_cen - opp_out)**2, 1))
        opp_cen = nodes_numpy_array_cen[n, :]


        #Obtain values from thickness, MTR, T2star
        #TODO: Separate to interface
        central_surface = nib.load(central_ls_file).get_fdata()
        outer_surface = nib.load(outer_ls_file).get_fdata()
        inner_surface = nib.load(inner_ls_file).get_fdata()
        thickness = np.abs(central_surface-outer_surface) + np.abs(central_surface-inner_surface)

        MTR = nib.load(self.inputs.mtr_file).get_fdata()
        T2star = nib.load(self.inputs.t2star_file).get_fdata()

        T2star_interp = RegularGridInterpolator((np.arange(0,T2star.shape[0]), np.arange(0,T2star.shape[1]), np.arange(0,T2star.shape[2])), T2star)
        T2star_opp = T2star_interp(opp_cen)
        T2star_coord = T2star_interp(coord_cen)

        MTR_interp = RegularGridInterpolator((np.arange(0,MTR.shape[0]), np.arange(0,MTR.shape[1]), np.arange(0,MTR.shape[2])), MTR)
        MTR_opp = MTR_interp(opp_cen)
        MTR_coord = MTR_interp(coord_cen)

        thickness_interp = RegularGridInterpolator((np.arange(0,thickness.shape[0]), np.arange(0,thickness.shape[1]), np.arange(0,thickness.shape[2])), thickness)
        thickness_opp = thickness_interp(opp_cen)
        thickness_coord = thickness_interp(coord_cen)

        outfile = self.inputs.prefix + '_lme.csv'

        with open(outfile, mode='w') as file:
            file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(['LMECoordinateMNI', 'LMECentralSurfCoordinateMNI', 'OppositeCentralSurfCoordinateMNI','Thickness_LME','Thickness_Opp','MTR_LME', 'MTR_Opp','T2Star_LME','T2Star_Opp'])
            file_writer.writerow([coord, coord_cen, opp_cen, thickness_coord[0], thickness_opp[0], MTR_coord[0], MTR_opp[0], T2star_coord[0], T2star_opp[0]])

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['lme_csv'] = os.path.abspath(self.inputs.prefix + '_lme.csv')

        return outputs