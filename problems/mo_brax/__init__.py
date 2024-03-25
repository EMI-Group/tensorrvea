from brax.envs import register_environment
from problems.mo_brax import mo_half_cheetah, mo_swimmer, mo_hopper, mo_hopper_m3

register_environment("mo_halfcheetah", mo_half_cheetah.MoHalfcheetah)
register_environment("mo_hopper_m2", mo_hopper.MoHopper)
register_environment("mo_hopper_m3", mo_hopper_m3.MoHopper)
register_environment("mo_swimmer", mo_swimmer.MoSwimmer)

