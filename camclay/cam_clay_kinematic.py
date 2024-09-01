import jax
from jax import numpy as jnp
import chex
import tempfile
from typing import Union, Dict, List, Tuple
import numpy as np
import os


@chex.dataclass
class MidState:
    stress: jnp.ndarray
    stiffness: jnp.ndarray
    strain: jnp.ndarray
    plastic_v: jnp.ndarray
    structure: jnp.ndarray
    harbeta: jnp.ndarray


@chex.dataclass
class SoilParameters:
    poisson_rate: Union[float, jnp.ndarray]
    void_rate: Union[float, jnp.ndarray]
    uppercase_mu: Union[float, jnp.ndarray]
    compression_index: Union[float, jnp.ndarray]
    swelling_index: Union[float, jnp.ndarray]
    overconsolidation: Union[float, jnp.ndarray]
    ocr_dissipation: Union[float, jnp.ndarray]
    structure: Union[float, jnp.ndarray]
    structure_dissipation: Union[float, jnp.ndarray]
    anisotropy: Union[float, jnp.ndarray]
    anisotropy_index: Union[float, jnp.ndarray]
    # for calculate
    axial_diff_strain: Union[float, jnp.ndarray]
    ymf: Union[float, jnp.ndarray]
    hee: Union[float, jnp.ndarray]
    # for start state
    sigma0: Union[float, jnp.ndarray]
    strain_max: Union[float, jnp.ndarray]

    def slice(self, indices: Union[int, List[int]]) -> 'SoilParameters':
        return SoilParameters(
            poisson_rate=self.poisson_rate[indices],
            void_rate=self.void_rate[indices],
            uppercase_mu=self.uppercase_mu[indices],
            compression_index=self.compression_index[indices],
            swelling_index=self.swelling_index[indices],
            overconsolidation=self.overconsolidation[indices],
            ocr_dissipation=self.ocr_dissipation[indices],
            structure_dissipation=self.structure_dissipation[indices],
            structure=self.structure[indices],
            anisotropy=self.anisotropy[indices],
            anisotropy_index=self.anisotropy_index[indices],
            axial_diff_strain=self.axial_diff_strain[indices],
            ymf=self.ymf[indices],
            hee=self.hee[indices],
            sigma0=self.sigma0[indices],
            strain_max=self.strain_max[indices]
        )

    def parallel_device(self):
        num_devices = jax.local_device_count()

        def split_across_devices(soil_params: SoilParameters) -> SoilParameters:
            fields = soil_params.__annotations__.keys()
            split_dict = {}
            for field in fields:
                field_value = getattr(soil_params, field)
                split_dict[field] = jnp.stack(jnp.array_split(field_value, num_devices))
            return SoilParameters(**split_dict)

        return split_across_devices(self)

    def show(self):
        fields_to_show = [
            "poisson_rate",
            "void_rate",
            "uppercase_mu",
            "compression_index",
            "swelling_index",
            "overconsolidation",
            "ocr_dissipation",
            "structure",
            "structure_dissipation",
            "anisotropy",
            "anisotropy_index",
        ]
        for field in fields_to_show:
            value = getattr(self, field)
            print(f"{field}: {value}")


# @jax.jit


def loading(carry: MidState, _, parameters: SoilParameters):
    # stress update

    _index = jnp.where(carry.stiffness[1, 1] + carry.stiffness[1, 2] == 0,
                       0, - carry.stiffness[1, 0] / (carry.stiffness[1, 1] + carry.stiffness[1, 2]))
    _load_index = jnp.array([1, _index, _index])
    _diff_strain = jnp.zeros(6)
    _diff_strain = _diff_strain.at[:3].set(parameters.axial_diff_strain * _load_index)
    _diff_stress = jnp.dot(carry.stiffness, _diff_strain)
    _stress = carry.stress + _diff_stress
    _strain = carry.strain + _diff_strain
    _hydrostatic = jnp.sum(_stress[:3]) / 3
    _sij = _hydrostatic * jnp.array([1, 1, 1, 0, 0, 0])
    _deviatoric = jnp.sqrt(jnp.sum(((_stress - _sij) ** 2) * jnp.array([0.5, 0.5, 0.5, 1, 1, 1])))

    def stress_val(stress_1, stress_2):
        return jnp.sqrt(jnp.sum(stress_1 * stress_2 * 2 * jnp.array([0.5, 0.5, 0.5, 1, 1, 1])))

    # n_ij=s_ij/sigma_m
    _eta = (_stress - _sij) / _hydrostatic
    # eta=sqrt(3/2*nij**2)
    _eta_val = jnp.sqrt(1.5) * stress_val(_eta, _eta)
    # anisotropic βij
    _har_beta = carry.harbeta
    _zzeta = jnp.sqrt(1.5) * stress_val(_har_beta, _har_beta)
    _eta_star = _eta - _har_beta
    _eta_star_val = jnp.sqrt(1.5) * stress_val(_eta_star, _eta_star)
    _ebs = 2 * _eta_star * _har_beta * jnp.array([0.5, 0.5, 0.5, 1, 1, 1])
    _zmb = 0.95
    _zbo = _zmb * jnp.sqrt(1.5) * parameters.ymf
    _etabb = _zbo * jnp.sqrt(1.5) * _eta_star / _eta_star_val - _har_beta
    _etabb = jnp.where(_etabb < 0, 0, _etabb)
    _etabb_val = stress_val(_etabb, _etabb)
    _setab = stress_val(_eta_star, _etabb) ** 2
    _sbetab = stress_val(_har_beta, _etabb) ** 2

    _stiffness = jnp.zeros([6, 6])
    _e0 = 3 * (1 - 2 * parameters.poisson_rate) * (1 + parameters.void_rate) * _hydrostatic / parameters.swelling_index
    _dmu = _e0 / 2 / (1 + parameters.poisson_rate)
    _dlam = _e0 * parameters.poisson_rate / (1 + parameters.poisson_rate) / (1 - 2 * parameters.poisson_rate)
    _stiffness = _stiffness.at[:3, :3].set(
        jnp.ones([3, 3]) * _dlam
    )
    _stiffness = _stiffness.at[:3, :3].set(
        _stiffness[:3, :3] + jnp.eye(3) * _dmu
    )
    _stiffness = _stiffness + jnp.eye(6) * _dmu
    _pm = (_hydrostatic * (parameters.ymf ** 2 - _zzeta ** 2 + _eta_star_val ** 2) /
           (parameters.ymf ** 2 - _zzeta ** 2))
    _pm_start = 98 * jnp.exp(carry.plastic_v / parameters.hee)
    _pm_bar = _pm_start / carry.structure
    _zr = _pm / _pm_bar

    _tt1 = 1 / (parameters.ymf ** 2 - _zzeta ** 2 + _eta_star_val ** 2) / _hydrostatic
    _tt2 = parameters.ymf ** 2 - _eta_val ** 2

    _f = jnp.zeros(6)
    _f = _f.at[:3].set((3 * _eta_star[:3] + _tt2 / 3) * _tt1)
    _f = _f.at[-3:].set(6 * _tt1 * _eta_star[-3:])
    _hdd = _tt1 * _tt2
    _ss1 = parameters.ymf ** 2 * _setab - _eta_star_val ** 2 * _sbetab - _zzeta ** 2 * _setab
    _strstd = (_hydrostatic / 98) ** 2 + 0.01
    _tms = (_tt2 + (6.0 * parameters.ymf * parameters.anisotropy_index * (1.0 - (_eta_val / parameters.ymf)) * (
            _zmb * parameters.ymf - _zzeta) * _ss1 * _eta_star_val / (
                     parameters.ymf ** 2 - _zzeta ** 2)) / (
                    parameters.ymf ** 2 - _zzeta ** 2 + _eta_star_val ** 2)
            - 2.0 * parameters.structure_dissipation * parameters.ymf * (
                        1.0 - carry.structure) * _eta_star_val - parameters.ocr_dissipation * parameters.ymf * jnp.log(
                _zr) / _zr * jnp.sqrt(6.0 * _eta_star_val ** 2 + _tt2 ** 2 / 3.0) * (_strstd / (_strstd + 1.0)))
    _hhh = _tms * _tt1 / parameters.hee
    _d = 1 / (_dlam * (_hdd ** 2) + _dmu * (
            jnp.sum(_f ** 2) + jnp.sum(_f[:3] ** 2)) + _hhh)

    _ramda1 = 2 * _dmu * _f[:3] + _dlam * jnp.sum(_f[:3])
    _ramda = jnp.dot(_ramda1, _diff_strain[:3]) + _dmu * jnp.dot(_f[-3:], _diff_strain[-3:])
    _plastic_v = carry.plastic_v + _ramda * _d * jnp.sum(_f[:3])
    old_state = MidState(stiffness=_stiffness, stress=_stress, strain=_strain,
                         plastic_v=carry.plastic_v, structure=carry.structure, harbeta=carry.harbeta)

    # _ww = _f * _dmu * jnp.array([2, 2, 2, 1, 1, 1]) + _dlam * jnp.sum(_f[:3]) * jnp.array([1, 1, 1, 0, 0, 0])
    # _stiffness2 = _stiffness - _d * jnp.outer(_ww, _ww)
    # new_carry = MidState(stiffness=_stiffness2, stress=_stress, strain=_strain, plastic_v=_plastic_v)
    def field_update(_stiffness) -> MidState:
        _ww = _f * _dmu * jnp.array([2, 2, 2, 1, 1, 1]) + _dlam * _hdd * jnp.array([1, 1, 1, 0, 0, 0])
        _stiffness2 = _stiffness - _d * jnp.outer(_ww, _ww)
        _drs = (2 * parameters.ymf * parameters.structure_dissipation * carry.structure
                * (1 - carry.structure) * _eta_star_val) * _ramda * _d * _tt1 / parameters.hee
        _tt3 = (2 * parameters.ymf * parameters.anisotropy_index * _tt1 * _ramda * _d / parameters.hee * _eta_star_val *
                _zmb * parameters.ymf - _zzeta)
        har_beta = carry.harbeta + _tt3 * _etabb
        new_state = MidState(stiffness=_stiffness2, stress=_stress, strain=_strain,
                             plastic_v=_plastic_v, structure=carry.structure+_drs, harbeta=har_beta)
        return new_state

    def field_condition(_sj, ramda):
        return jax.lax.cond((_sj != 0) & (ramda > 0), lambda x: True, lambda x: False, None)

    new_carry = jax.lax.cond(field_condition(_deviatoric, _ramda * _d),
                             field_update, lambda *args: old_state, _stiffness)

    return new_carry, [new_carry.stress[0] - new_carry.stress[1], new_carry.strain[:3]]


# @jax.jit
def initialize(parameters: SoilParameters) -> MidState:
    _stress = jnp.zeros(6)
    _stress = _stress.at[:3].set(parameters.sigma0)
    _strain = jnp.zeros(6)
    _stiffness = jnp.zeros((6, 6))
    _hydrostatic = jnp.sum(_stress[:3]) / 3
    _sij = _hydrostatic * jnp.array([1, 1, 1, 0, 0, 0])
    _deviatoric = jnp.sqrt(jnp.sum(((_stress - _sij) ** 2) * jnp.array([0.5, 0.5, 0.5, 1, 1, 1])))
    _e0 = 3 * (1 - 2 * parameters.poisson_rate) * (1 + parameters.void_rate) * _hydrostatic / parameters.swelling_index
    _dmu = _e0 / 2 / (1 + parameters.poisson_rate)
    _dlam = _e0 * parameters.poisson_rate / (1 + parameters.poisson_rate) / (1 - 2 * parameters.poisson_rate)
    _stiffness = _stiffness.at[:3, :3].set(
        jnp.ones([3, 3]) * _dlam
    )
    _stiffness = _stiffness.at[:3, :3].set(
        _stiffness[:3, :3] + jnp.eye(3) * _dmu
    )
    _stiffness = _stiffness + jnp.eye(6) * _dmu
    _harbeta = parameters.anisotropy * jnp.array([2/3, -1./3, -1/3, 0, 0, 0])
    pm0 = parameters.sigma0 * parameters.ymf ** 2 / (parameters.ymf ** 2 - parameters.anisotropy ** 2)
    _plastic_v = parameters.hee * jnp.log(parameters.overconsolidation * pm0 * parameters.structure / 98)

    return MidState(stress=_stress, stiffness=_stiffness, strain=_strain,
                    plastic_v=_plastic_v, structure=parameters.structure, harbeta=_harbeta)


def create_save_simulation(configs, config_name="default", sigma0=98, chunk_size=50000) -> List[
    Tuple[str, Tuple[int]]]:
    if jax.local_device_count() > 2:
        chunk_size = 40000
    elif jax.local_device_count() < 2:
        chunk_size = 5000

    config = configs.get(config_name, configs["default"])
    parameter_ranges = {
        "poisson_rate": np.linspace,
        "void_rate": np.linspace,
        "uppercase_mu": np.linspace,
        "compression_index": np.linspace,
        "swelling_index": np.linspace,
        "overconsolidation": np.linspace,
        "ocr_dissipation": np.linspace,
        "ocr_index": np.linspace,
        "structure": jnp.linspace,
        "structure_dissipation": jnp.linspace,
        "anisotropy": jnp.linspace,
        "anisotropy_index": jnp.linspace,
    }
    parameter_values = {key: func(config[key][0], config[key][1], config[key][2]) for key, func in
                        parameter_ranges.items()}
    grids = np.meshgrid(*parameter_values.values(), indexing='ij')
    raveled_params = {key: grid.ravel() for key, grid in zip(parameter_ranges.keys(), grids)}

    def chunk_data(data, chunk_size):
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    chunks = {key: chunk_data(raveled_params[key], chunk_size) for key in raveled_params}

    file_paths = []
    dtype = np.dtype([
        ('poisson_rate', np.float64),
        ('void_rate', np.float64),
        ('uppercase_mu', np.float64),
        ('compression_index', np.float64),
        ('swelling_index', np.float64),
        ('overconsolidation', np.float64),
        ('ocr_dissipation', np.float64),
        ('structure', np.float64),
        ('structure_dissipation', np.float64),
        ('anisotropy', np.float64),
        ('anisotropy_index', np.float64),
        ('axial_diff_strain', np.float64),
        ('hee', np.float64),
        ('ymf', np.float64),
        ('sigma0', np.float64),
        ('strain_max', np.float64)
    ])

    for i in range(len(chunks['poisson_rate'])):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mmap') as temp_file:
            ocr_p = 1
            if sigma0 == 100:
                ocr_p = 1
            elif sigma0 == 200:
                ocr_p = 1 - chunks['ocr_index'][i]
            elif sigma0 == 50:
                ocr_p = 1 + chunks['ocr_index'][i]
            shape = chunks['poisson_rate'][i].shape
            memmap = np.memmap(temp_file.name, dtype=dtype, mode='w+', shape=shape)
            memmap['poisson_rate'] = chunks['poisson_rate'][i]
            memmap['void_rate'] = chunks['void_rate'][i]
            memmap['uppercase_mu'] = chunks['uppercase_mu'][i]
            memmap['compression_index'] = chunks['compression_index'][i]
            memmap['swelling_index'] = chunks['swelling_index'][i]
            memmap['overconsolidation'] = ocr_p * chunks['overconsolidation'][i]
            memmap['ocr_dissipation'] = chunks['ocr_dissipation'][i]
            memmap['structure'] = chunks['structure'][i]
            memmap['structure_dissipation'] = chunks['structure_dissipation'][i]
            memmap['anisotropy'] = chunks['anisotropy'][i]
            memmap['anisotropy_index'] = chunks['anisotropy_index'][i]
            memmap['axial_diff_strain'] = np.full(shape, 0.16 / 10000)
            memmap['hee'] = (chunks['compression_index'][i] - chunks['swelling_index'][i]) / (
                    1 + chunks['void_rate'][i])
            memmap['ymf'] = 3 * (chunks['uppercase_mu'][i] - 1.0) / (chunks['uppercase_mu'][i] + 2.0)
            memmap['sigma0'] = np.full(shape, sigma0)
            memmap['strain_max'] = np.full(shape, 0.16)
            memmap.flush()
            file_paths.append((temp_file.name, shape))

    return file_paths


def load_chunk(temp_file: str, shape: Tuple[int]) -> SoilParameters:
    dtype = np.dtype([
        ('poisson_rate', np.float64),
        ('void_rate', np.float64),
        ('uppercase_mu', np.float64),
        ('compression_index', np.float64),
        ('swelling_index', np.float64),
        ('overconsolidation', np.float64),
        ('ocr_dissipation', np.float64),
        ('structure', np.float64),
        ('structure_dissipation', np.float64),
        ('anisotropy', np.float64),
        ('anisotropy_index', np.float64),
        ('axial_diff_strain', np.float64),
        ('hee', np.float64),
        ('ymf', np.float64),
        ('sigma0', np.float64),
        ('strain_max', np.float64)
    ])

    memmap = np.memmap(temp_file, dtype=dtype, mode='r', shape=shape)
    chunk_params = SoilParameters(
        poisson_rate=jnp.array(memmap['poisson_rate']),
        void_rate=jnp.array(memmap['void_rate']),
        uppercase_mu=jnp.array(memmap['uppercase_mu']),
        compression_index=jnp.array(memmap['compression_index']),
        swelling_index=jnp.array(memmap['swelling_index']),
        overconsolidation=jnp.array(memmap['overconsolidation']),
        ocr_dissipation=jnp.array(memmap['ocr_dissipation']),
        structure=jnp.array(memmap['structure']),
        structure_dissipation=jnp.array(memmap['structure_dissipation']),
        anisotropy=jnp.array(memmap['anisotropy']),
        anisotropy_index=jnp.array(memmap['anisotropy_index']),
        axial_diff_strain=jnp.array(memmap['axial_diff_strain']),
        hee=jnp.array(memmap['hee']),
        ymf=jnp.array(memmap['ymf']),
        sigma0=jnp.array(memmap['sigma0']),
        strain_max=jnp.array(memmap['strain_max'])
    )
    return chunk_params


def cleanup_temp_files(temp_files):
    for temp_file, _ in temp_files:
        try:
            os.remove(temp_file)
            # print(f"Temporary file {temp_file} deleted.")
        except OSError as e:
            print(f"Error deleting temporary file {temp_file}: {e}")


def simulation(parameters: SoilParameters):
    init = initialize(parameters)
    _, all_states = jax.lax.scan(lambda state, x: loading(state, x, parameters), init, None, length=10000)
    return all_states


def parallel_device(soil_params_chunks: List[SoilParameters]) -> List[SoilParameters]:
    num_devices = jax.local_device_count()

    # num_devices = 1
    # print(f"num_devices: {num_devices}")

    def split_across_devices(soil_params: SoilParameters) -> SoilParameters:
        fields = soil_params.__annotations__.keys()
        split_dict = {}
        for field in fields:
            field_value = getattr(soil_params, field)
            split_dict[field] = jnp.stack(jnp.array_split(field_value, num_devices))
        return SoilParameters(**split_dict)

    return [split_across_devices(chunk) for chunk in soil_params_chunks]


# def single_simulation(soil_params_chunks: List[SoilParameters]) -> SoilParameters:
#     for i, chunk_params in enumerate(soil_params_chunks):
#         stress, _ = simulation(chunk_params)
#     return stress


if __name__ == "__main__":
    jax.config.update('jax_enable_x64', True)
    batchsize = 10000
    configs: Dict[str, Dict[str, Union[float, tuple, List[int]]]] = {
        "default": {
            "poisson_rate": (0, 0.1, 1),
            "void_rate": (0.88, 0.88, 1),
            "uppercase_mu": (3.8, 4.0, 10),
            "compression_index": (0.0955, 0.1, 2),
            "swelling_index": (0.00884, 0.009, 10),
            "overconsolidation": (1, 20, 10),
            "ocr_dissipation": (100, 200, 10),
            "ocr_index": (0, 0.2, 4),
            "structure": (1, 1, 1),
            "structure_dissipation": (0.38, 0.38, 1),
            "anisotropy": (0, 1, 1),
            "anisotropy_index": (1.5, 1.5, 1),
        },
    }
    vectorized_simulation = jax.pmap(jax.vmap(simulation))
    _params = create_save_simulation(configs)
    for file_path, shape in _params:
        chunk_params = load_chunk(file_path, shape=shape)
        _stress, _ = vectorized_simulation(chunk_params.parallel_device())
        # parameters = chunk_params.slice(0)
        # carry = initialize(parameters)
        # for i in range(100):
        #     carry, results = loading(carry, 0, parameters)
        #     print(results)
        print(_stress)
        break
