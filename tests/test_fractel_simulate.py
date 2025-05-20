from unittest.mock import MagicMock
import numpy as np
import pytest
from fractel.fractel import run_simulation

   
def test_run_simulation_with_valid_args():
    args = MagicMock()
    args.num_guides = [10, 20]
    args.num_simulations = 100
    args.bnd = 0.5
    args.bnd_min = 3

    sim_dict = run_simulation(args)

    assert isinstance(sim_dict, dict)
    assert len(sim_dict) == len(args.num_guides)
    for num_guides, simulations in sim_dict.items():
        assert num_guides in args.num_guides
        assert isinstance(simulations, np.ndarray)
        assert simulations.shape == (args.num_simulations,)

def test_run_simulation_with_fractional_bnd():
    args = MagicMock()
    args.num_guides = [10]
    args.num_simulations = 50
    args.bnd = 0.3  # Fractional bound
    args.bnd_min = 2

    sim_dict = run_simulation(args)

    assert isinstance(sim_dict, dict)
    assert len(sim_dict) == len(args.num_guides)
    for num_guides, simulations in sim_dict.items():
        assert num_guides in args.num_guides
        assert isinstance(simulations, np.ndarray)
        assert simulations.shape == (args.num_simulations,)

def test_run_simulation_with_large_bnd():
    args = MagicMock()
    args.num_guides = [10]
    args.num_simulations = 50
    args.bnd = 15  # Larger than the number of guides
    args.bnd_min = 3

    sim_dict = run_simulation(args)

    assert isinstance(sim_dict, dict)
    assert len(sim_dict) == len(args.num_guides)
    for num_guides, simulations in sim_dict.items():
        assert num_guides in args.num_guides
        assert isinstance(simulations, np.ndarray)
        assert simulations.shape == (args.num_simulations,)

def test_run_simulation_with_small_bnd_min():
    args = MagicMock()
    args.num_guides = [10]
    args.num_simulations = 50
    args.bnd = 0.5
    args.bnd_min = 1  # Small minimum bound

    sim_dict = run_simulation(args)

    assert isinstance(sim_dict, dict)
    assert len(sim_dict) == len(args.num_guides)
    for num_guides, simulations in sim_dict.items():
        assert num_guides in args.num_guides
        assert isinstance(simulations, np.ndarray)
        assert simulations.shape == (args.num_simulations,)

def test_run_simulation_with_invalid_num_guides():
    args = MagicMock()
    args.num_guides = []
    args.num_simulations = 50
    args.bnd = 0.5
    args.bnd_min = 3

    with pytest.raises(ValueError):
        run_simulation(args)

def test_run_simulation_with_zero_simulations():
    args = MagicMock()
    args.num_guides = [10]
    args.num_simulations = 0  # No simulations
    args.bnd = 0.5
    args.bnd_min = 3

    sim_dict = run_simulation(args)

    assert isinstance(sim_dict, dict)
    assert len(sim_dict) == len(args.num_guides)
    for num_guides, simulations in sim_dict.items():
        assert num_guides in args.num_guides
        assert isinstance(simulations, np.ndarray)
        assert simulations.size == 0
