import numpy as np
from pyscf import gto, dft, scf, grad
import rt_integrators
import rt_observables
import rt_output
import rt_cap
import rt_vapp
import rt_nuclei
import ehrenfest_force
import rt_cpa
#from ehrenfest_brute_force import EhrenfestBruteForce
from basis_utils import translatebasis

'''
Real-time SCF + Ehrenfest
'''

class rt_ehrenfest:
    def __init__(self, rt_scf, Ne_step=1, N_step=1):
        self._scf = rt_scf._scf
        ######################
        self._scf.mo_energy *= 0.
        ######################
        self.Ne_step = Ne_step
        self.N_step = N_step
        self.timestep = rt_scf.timestep
        self.frequency = rt_scf.frequency
        self.total_steps = rt_scf.total_steps
        self.filename = rt_scf.filename
        self.occ = rt_scf.occ
        self.prop = rt_scf.prop
        self.ovlp = rt_scf.ovlp
        self.orth = rt_scf.orth

        self.nuc = rt_nuclei.rt_nuc(self._scf.mol)
        if self._scf.mo_coeff.dtype != np.complex128:
            self._scf.mo_coeff = self._scf.mo_coeff.astype(np.complex128)
        self.set_grad()

        self.den_ao = self._scf.make_rdm1(mo_occ = self.occ)
        self.t = 0
        rt_observables.init_observables(self)

    def set_grad(self):
        if self._scf.istype('RKS'):
            self._grad = self._scf.apply(grad.RKS)
        elif self._scf.istype('RHF'):
            self._grad = self._scf.apply(grad.RHF)
        elif self._scf.istype('UKS'):
            self._grad = self._scf.apply(grad.UKS)
        elif self._scf.istype('UHF'):
            self._grad = self._scf.apply(grad.UHF)
        elif self._scf.istype('GKS'):
            self._grad = self._scf.apply(grad.GKS)
        elif self._scf.istype('GHF'):
            self._grad = self._scf.apply(grad.GHF)
        else:
            raise Exception('unknown scf method')

    def get_force(self):
        self.set_grad()
        self.nuc.force = ehrenfest_force.get_force(self._grad)
        #EBF = EhrenfestBruteForce(self.rt_scf._scf, self.nuc)
        #self.nuc.force = EBF.get_brute_force()

    def get_fock_orth(self, den_ao):
        #self.fock = self._scf.get_fock(dm=den_ao) + rt_cpa.get_fock_pert(self)
        self.fock = self._scf.get_fock(dm=den_ao)
        return np.matmul(self.orth.T, np.matmul(self.fock, self.orth))

    def kernel(self):
        #rt_output.create_output_file(self.rt_scf)
        self.temp_create_output_file()
        #rt_observables.prepare_observables(self.rt_scf)
        mo_coeff_old = self._scf.mo_coeff

        for i in range(0, self.total_steps):

            if np.mod(i, self.frequency) == 0:
                # rt_observables.get_observables(self.rt_scf, mo_coeff_print)
                self.temp_update_output_file()
                self.h1e = scf.hf.get_hcore(self._scf.mol)
                self.vhf = scf.hf.get_veff(self._scf.mol, self.den_ao)
                print(scf.hf.energy_tot(self._scf, dm=self.den_ao, h1e=self.h1e, vhf=self.vhf) + self.nuc.get_ke())
                #print('---------------------------------------------\n')
                #print(self.den_ao)
                #print('---------------------------------------------\n')

            self.nuc.get_vel(self.timestep)
            self.nuc.get_pos(self.timestep)
            mo_coeff_old = rt_integrators.magnus_step(self, mo_coeff_old)
            self._scf.mol = self.nuc.get_mol()
            self.ovlp = self._scf.mol.intor_symmetric("int1e_ovlp")
            self.orth = scf.addons.canonical_orth_(self.ovlp)
            self.get_force()
            self.nuc.get_pos(self.timestep)
            self.nuc.get_vel(self.timestep)
            self.t += self.timestep

        #rt_observables.get_observables(rt_mf, mo_coeff_print)

        return self

    def temp_create_output_file(self):
        pos_file = open(F'{self.filename}' + '_pos.txt','w')
        vel_file = open(F'{self.filename}' + '_vel.txt','w')
        force_file = open(F'{self.filename}' + '_force.txt','w')

        pos_file.close()
        vel_file.close()
        force_file.close()

    def temp_update_output_file(self):
        pos_file = open(F'{self.filename}' + '_pos.txt','a')
        vel_file = open(F'{self.filename}' + '_vel.txt','a')
        force_file = open(F'{self.filename}' + '_force.txt','a')
        np.savetxt(pos_file, self.nuc.pos, '%20.8e')
        pos_file.write('\n')
        np.savetxt(vel_file, self.nuc.vel, '%20.8e')
        vel_file.write('\n')
        np.savetxt(force_file, self.nuc.force, '%20.8e')
        force_file.write('\n')
        force_file.write(f'{self._scf.energy_tot() + self.nuc.get_ke()}')
        force_file.write('\n')
        pos_file.close()
        vel_file.close()
        force_file.close()
