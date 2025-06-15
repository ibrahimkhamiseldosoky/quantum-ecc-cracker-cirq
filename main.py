

import time
import json
import matplotlib.pyplot as plt
import numpy as np
import cirq
import platform
import psutil
from typing import Dict, List, Tuple
import os
from collections import Counter

# Hardware detection
def get_hardware_info() -> Dict:
    """Get system hardware information"""
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'python_version': platform.python_version(),
        'cirq_version': cirq.__version__
    }
    
    # Try to detect GPU
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            info['gpu'] = [{'name': gpu.name, 'memory_mb': gpu.memoryTotal} for gpu in gpus]
        else:
            info['gpu'] = 'None detected'
    except ImportError:
        info['gpu'] = 'GPUtil not available'
    
    return info

class ToyECCSystem:
    """Simple ECC implementation for demonstration"""
    
    def __init__(self, a: int, b: int, p: int):
        self.a = a
        self.b = b
        self.p = p
        self.points = self._generate_points()
    
    def _generate_points(self) -> List[Tuple]:
        """Generate all points on the elliptic curve"""
        points = [(None, None)]  # Point at infinity
        
        for x in range(self.p):
            y_squared = (x**3 + self.a * x + self.b) % self.p
            for y in range(self.p):
                if (y * y) % self.p == y_squared:
                    points.append((x, y))
        
        return points
    
    def find_generator(self) -> Tuple:
        """Find a generator point for the group"""
        for point in self.points[1:]:  # Skip point at infinity
            if point != (None, None):
                return point
        return None
    
    def point_multiply(self, k: int, point: Tuple) -> Tuple:
        """Scalar multiplication of point by k"""
        if point == (None, None) or k == 0:
            return (None, None)
        
        if k == 1:
            return point
        
        # Simple double-and-add algorithm
        result = (None, None)
        addend = point
        
        while k:
            if k & 1:
                result = self.point_add(result, addend)
            addend = self.point_add(addend, addend)
            k >>= 1
        
        return result
    
    def point_add(self, p1: Tuple, p2: Tuple) -> Tuple:
        """Add two points on the elliptic curve"""
        if p1 == (None, None):
            return p2
        if p2 == (None, None):
            return p1
        
        x1, y1 = p1
        x2, y2 = p2
        
        if x1 == x2:
            if y1 == y2:
                # Point doubling
                s = (3 * x1 * x1 + self.a) * pow(2 * y1, -1, self.p) % self.p
            else:
                return (None, None)  # Points are inverses
        else:
            # Point addition
            s = (y2 - y1) * pow(x2 - x1, -1, self.p) % self.p
        
        x3 = (s * s - x1 - x2) % self.p
        y3 = (s * (x1 - x3) - y1) % self.p
        
        return (x3, y3)

class QuantumECDLPSolver:
    """Quantum solver for the Elliptic Curve Discrete Logarithm Problem using Cirq"""
    
    def __init__(self, curve: ToyECCSystem, generator: Tuple, public_key: Tuple):
        self.curve = curve
        self.generator = generator
        self.public_key = public_key
        self.field_size = curve.p
        
        # Determine number of qubits needed
        self.num_qubits = max(4, int(np.ceil(np.log2(self.field_size))) + 2)
        self.qubits = cirq.LineQubit.range(self.num_qubits)
    
    def solve_classical_bruteforce(self) -> int:
        """Classical brute force solution for comparison"""
        for k in range(1, self.field_size):
            if self.curve.point_multiply(k, self.generator) == self.public_key:
                return k
        return None
    
    def create_quantum_oracle(self) -> cirq.Circuit:
        """Create quantum oracle circuit for the ECDLP"""
        circuit = cirq.Circuit()
        
        # Create superposition of all possible private keys
        for i in range(int(np.ceil(np.log2(self.field_size)))):
            circuit.append(cirq.H(self.qubits[i]))
        
        # Add oracle marking (simplified for toy examples)
        # In practice, this would implement modular exponentiation
        circuit.append(cirq.X(self.qubits[-1]))
        
        # Simplified phase oracle
        for i in range(len(self.qubits) - 1):
            circuit.append(cirq.CZ(self.qubits[i], self.qubits[-1]))
        
        return circuit
    
    def create_diffusion_operator(self) -> cirq.Circuit:
        """Create Grover diffusion operator"""
        circuit = cirq.Circuit()
        
        # Apply Hadamard gates
        for i in range(int(np.ceil(np.log2(self.field_size)))):
            circuit.append(cirq.H(self.qubits[i]))
        
        # Apply controlled-Z gate with all qubits as controls
        for i in range(int(np.ceil(np.log2(self.field_size)))):
            circuit.append(cirq.X(self.qubits[i]))
        
        # Multi-controlled Z gate (simplified)
        if len(self.qubits) > 2:
            circuit.append(cirq.CCZ(self.qubits[0], self.qubits[1], self.qubits[2]))
        
        for i in range(int(np.ceil(np.log2(self.field_size)))):
            circuit.append(cirq.X(self.qubits[i]))
        
        # Apply Hadamard gates again
        for i in range(int(np.ceil(np.log2(self.field_size)))):
            circuit.append(cirq.H(self.qubits[i]))
        
        return circuit
    
    def create_shor_like_circuit(self) -> cirq.Circuit:
        """Create a simplified Shor-like quantum circuit"""
        circuit = cirq.Circuit()
        
        # Initialize qubits in superposition
        for i in range(int(np.ceil(np.log2(self.field_size)))):
            circuit.append(cirq.H(self.qubits[i]))
        
        # Add quantum oracle
        oracle = self.create_quantum_oracle()
        circuit += oracle
        
        # Apply Grover-like amplification
        iterations = max(1, int(np.sqrt(self.field_size) / 4))
        for _ in range(iterations):
            diffusion = self.create_diffusion_operator()
            circuit += diffusion
            circuit += oracle
        
        # Add measurements
        measurement_qubits = self.qubits[:int(np.ceil(np.log2(self.field_size)))]
        circuit.append(cirq.measure(*measurement_qubits, key='result'))
        
        return circuit
    
    def run_quantum_solver(self, shots: int = 1024) -> Dict:
        """Run the quantum ECDLP solver"""
        try:
            # Create quantum circuit
            circuit = self.create_shor_like_circuit()
            
            # Run simulation
            simulator = cirq.Simulator()
            result = simulator.run(circuit, repetitions=shots)
            
            # Analyze measurements
            measurements = result.measurements['result']
            counts = Counter()
            
            for measurement in measurements:
                # Convert binary measurement to integer
                value = 0
                for i, bit in enumerate(measurement):
                    value += bit * (2 ** i)
                counts[value] += 1
            
            # Find most likely result
            if counts:
                most_likely = max(counts.items(), key=lambda x: x[1])
                candidate_key = most_likely[0] % self.field_size
                
                # Verify the result
                if candidate_key > 0:
                    test_point = self.curve.point_multiply(candidate_key, self.generator)
                    success = (test_point == self.public_key)
                    probability = most_likely[1] / shots
                else:
                    success = False
                    probability = 0
                
                return {
                    'success': success,
                    'private_key': candidate_key if success else None,
                    'probability': probability,
                    'counts': dict(counts)
                }
            else:
                return {
                    'success': False,
                    'private_key': None,
                    'probability': 0,
                    'counts': {}
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'counts': {}
            }
    
    def analyze_circuit_complexity(self) -> Dict:
        """Analyze quantum circuit complexity"""
        circuit = self.create_shor_like_circuit()
        
        # Count gates by type
        gate_counts = Counter()
        for moment in circuit:
            for operation in moment:
                gate_counts[type(operation.gate).__name__] += 1
        
        return {
            'num_qubits': len(self.qubits),
            'depth': len(circuit),
            'num_gates': sum(gate_counts.values()),
            'gate_breakdown': dict(gate_counts)
        }

class QuantumECCExperiment:
    """
    Complete experimental framework for quantum ECC attacks using Cirq
    """
    
    def __init__(self):
        self.results = []
        self.hardware_info = get_hardware_info()
        print("Quantum ECC Attack Experiment Framework (Cirq)")
        print("=" * 50)
        self.print_hardware_info()
    
    def print_hardware_info(self):
        """Print hardware information"""
        print("\nHardware Configuration:")
        print(f"Platform: {self.hardware_info['platform']}")
        print(f"CPU: {self.hardware_info['processor']}")
        print(f"CPU Cores: {self.hardware_info['cpu_count']}")
        print(f"Memory: {self.hardware_info['memory_gb']} GB")
        print(f"GPU: {self.hardware_info['gpu']}")
        print(f"Python: {self.hardware_info['python_version']}")
        print(f"Cirq: {self.hardware_info['cirq_version']}")
    
    def run_single_experiment(self, curve_params: Tuple[int, int, int], 
                            private_key: int, shots: int = 1024) -> Dict:
        """
        Run a single quantum ECC attack experiment
        """
        a, b, p = curve_params
        
        print(f"\n--- Experiment: Curve y² = x³ + {a}x + {b} (mod {p}) ---")
        print(f"Target private key: {private_key}")
        
        try:
            # Setup ECC system
            curve = ToyECCSystem(a, b, p)
            generator = curve.find_generator()
            
            if generator is None:
                return {'success': False, 'error': 'No generator found'}
            
            # Generate key pair
            public_key = curve.point_multiply(private_key, generator)
            
            # Classical timing
            start_time = time.time()
            solver = QuantumECDLPSolver(curve, generator, public_key)
            classical_result = solver.solve_classical_bruteforce()
            classical_time = time.time() - start_time
            
            # Quantum timing
            start_time = time.time()
            quantum_result = solver.run_quantum_solver(shots=shots)
            quantum_time = time.time() - start_time
            
            # Circuit analysis
            circuit_stats = solver.analyze_circuit_complexity()
            
            # Compile results
            experiment_result = {
                'curve_params': curve_params,
                'field_size': p,
                'private_key': private_key,
                'public_key': public_key,
                'generator': generator,
                'classical_time': classical_time,
                'classical_result': classical_result,
                'quantum_time': quantum_time,
                'quantum_success': quantum_result['success'],
                'quantum_result': quantum_result.get('private_key', None),
                'quantum_probability': quantum_result.get('probability', 0),
                'circuit_qubits': circuit_stats['num_qubits'],
                'circuit_depth': circuit_stats['depth'],
                'circuit_gates': circuit_stats['num_gates'],
                'gate_breakdown': circuit_stats['gate_breakdown'],
                'shots': shots,
                'measurement_counts': quantum_result['counts']
            }
            
            return experiment_result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def run_parameter_sweep(self) -> List[Dict]:
        """
        Run experiments across different curve parameters and key sizes
        """
        print("\n" + "="*60)
        print("PARAMETER SWEEP EXPERIMENTS")
        print("="*60)
        
        # Test cases: (a, b, p, private_keys_to_test)
        test_cases = [
            (2, 3, 5, [2, 3]),      # Very small field
            (2, 3, 7, [2, 3, 4]),   # Small field
            (1, 6, 11, [3, 5, 7]),  # Medium field
            (4, 4, 13, [2, 6, 8]),  # Larger field
        ]
        
        all_results = []
        
        for a, b, p, private_keys in test_cases:
            print(f"\nTesting curve y² = x³ + {a}x + {b} (mod {p})")
            
            for private_key in private_keys:
                result = self.run_single_experiment((a, b, p), private_key, shots=2048)
                if result.get('success', True):  # Not failed due to error
                    all_results.append(result)
                    self.results.append(result)
                else:
                    print(f"  Failed: {result.get('error', 'Unknown error')}")
        
        return all_results
    
    def analyze_success_rates(self) -> Dict:
        """Analyze success rates across experiments"""
        if not self.results:
            return {}
        
        total_experiments = len(self.results)
        successful_quantum = sum(1 for r in self.results if r.get('quantum_success', False))
        successful_classical = sum(1 for r in self.results if r.get('classical_result') is not None)
        
        analysis = {
            'total_experiments': total_experiments,
            'quantum_success_rate': successful_quantum / total_experiments,
            'classical_success_rate': successful_classical / total_experiments,
            'average_quantum_time': np.mean([r['quantum_time'] for r in self.results]),
            'average_classical_time': np.mean([r['classical_time'] for r in self.results]),
            'average_quantum_probability': np.mean([r.get('quantum_probability', 0) for r in self.results]),
            'qubit_usage': [r['circuit_qubits'] for r in self.results],
            'circuit_depths': [r['circuit_depth'] for r in self.results],
            'field_sizes': [r['field_size'] for r in self.results]
        }
        
        return analysis
    
    def plot_results(self):
        """Create comprehensive plots of experimental results"""
        if not self.results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Quantum ECC Attack Analysis (Cirq Implementation)', fontsize=16)
        
        # 1. Success Rate Comparison
        ax1 = axes[0, 0]
        quantum_successes = [r.get('quantum_success', False) for r in self.results]
        classical_successes = [r.get('classical_result') is not None for r in self.results]
        
        methods = ['Quantum', 'Classical']
        success_rates = [
            sum(quantum_successes) / len(quantum_successes),
            sum(classical_successes) / len(classical_successes)
        ]
        
        bars = ax1.bar(methods, success_rates, color=['blue', 'red'], alpha=0.7)
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rate Comparison')
        ax1.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{rate:.2f}', ha='center', va='bottom')
        
        # 2. Timing Comparison
        ax2 = axes[0, 1]
        quantum_times = [r['quantum_time'] for r in self.results]
        classical_times = [r['classical_time'] for r in self.results]
        
        ax2.scatter(range(len(quantum_times)), quantum_times, 
                   label='Quantum', color='blue', alpha=0.7)
        ax2.scatter(range(len(classical_times)), classical_times, 
                   label='Classical', color='red', alpha=0.7)
        ax2.set_xlabel('Experiment Number')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Execution Time Comparison')
        ax2.legend()
        ax2.set_yscale('log')
        
        # 3. Quantum Probability Distribution
        ax3 = axes[0, 2]
        quantum_probs = [r.get('quantum_probability', 0) for r in self.results if r.get('quantum_success', False)]
        if quantum_probs:
            ax3.hist(quantum_probs, bins=10, alpha=0.7, color='green')
            ax3.set_xlabel('Success Probability')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Quantum Success Probability Distribution')
        else:
            ax3.text(0.5, 0.5, 'No Successful\nQuantum Results', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Quantum Success Probability Distribution')
        
        # 4. Circuit Complexity vs Field Size
        ax4 = axes[1, 0]
        field_sizes = [r['field_size'] for r in self.results]
        circuit_qubits = [r['circuit_qubits'] for r in self.results]
        circuit_depths = [r['circuit_depth'] for r in self.results]
        
        ax4.scatter(field_sizes, circuit_qubits, label='Qubits', alpha=0.7)
        ax4.scatter(field_sizes, circuit_depths, label='Depth', alpha=0.7)
        ax4.set_xlabel('Field Size (p)')
        ax4.set_ylabel('Circuit Complexity')
        ax4.set_title('Circuit Complexity vs Field Size')
        ax4.legend()
        
        # 5. Success Rate vs Field Size
        ax5 = axes[1, 1]
        field_success_data = {}
        for result in self.results:
            p = result['field_size']
            if p not in field_success_data:
                field_success_data[p] = {'total': 0, 'quantum_success': 0}
            field_success_data[p]['total'] += 1
            if result.get('quantum_success', False):
                field_success_data[p]['quantum_success'] += 1
        
        field_sizes_unique = sorted(field_success_data.keys())
        success_rates_by_field = [
            field_success_data[p]['quantum_success'] / field_success_data[p]['total']
            for p in field_sizes_unique
        ]
        
        ax5.plot(field_sizes_unique, success_rates_by_field, 'o-', color='purple')
        ax5.set_xlabel('Field Size (p)')
        ax5.set_ylabel('Quantum Success Rate')
        ax5.set_title('Success Rate vs Field Size')
        ax5.set_ylim(0, 1.1)
        
        # 6. Gate Type Distribution
        ax6 = axes[1, 2]
        all_gates = Counter()
        for result in self.results:
            gate_breakdown = result.get('gate_breakdown', {})
            for gate_type, count in gate_breakdown.items():
                all_gates[gate_type] += count
        
        if all_gates:
            gate_types = list(all_gates.keys())
            gate_counts = list(all_gates.values())
            
            ax6.pie(gate_counts, labels=gate_types, autopct='%1.1f%%')
            ax6.set_title('Gate Type Distribution')
        else:
            ax6.text(0.5, 0.5, 'No Gate Data\nAvailable', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Gate Type Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self) -> str:
        """Generate comprehensive experimental report"""
        if not self.results:
            return "No experimental results available."
        
        analysis = self.analyze_success_rates()
        
        report = f"""
# Quantum Elliptic Curve Discrete Logarithm Attack Report (Cirq)

## Hardware Configuration
- Platform: {self.hardware_info['platform']}
- CPU: {self.hardware_info['processor']} ({self.hardware_info['cpu_count']} cores)
- Memory: {self.hardware_info['memory_gb']} GB
- GPU: {self.hardware_info['gpu']}
- Cirq Version: {self.hardware_info['cirq_version']}

## Experimental Results

### Overall Performance
- Total Experiments: {analysis['total_experiments']}
- Quantum Success Rate: {analysis['quantum_success_rate']:.2%}
- Classical Success Rate: {analysis['classical_success_rate']:.2%}
- Average Quantum Time: {analysis['average_quantum_time']:.4f} seconds
- Average Classical Time: {analysis['average_classical_time']:.4f} seconds

### Quantum Circuit Characteristics
- Average Qubits Used: {np.mean(analysis['qubit_usage']):.1f}
- Average Circuit Depth: {np.mean(analysis['circuit_depths']):.1f}
- Field Sizes Tested: {min(analysis['field_sizes'])} - {max(analysis['field_sizes'])}

"""
        
        for i, result in enumerate(self.results):
            gate_info = result.get('gate_breakdown', {})
            gate_summary = ', '.join([f"{k}: {v}" for k, v in gate_info.items()])
            
            report += f"""
### Experiment {i+1}
- **Curve**: y² = x³ + {result['curve_params'][0]}x + {result['curve_params'][1]} (mod {result['curve_params'][2]})
- **Private Key**: {result['private_key']}
- **Quantum Success**: {'✓' if result.get('quantum_success', False) else '✗'}
- **Classical Time**: {result['classical_time']:.4f}s
- **Quantum Time**: {result['quantum_time']:.4f}s
- **Circuit Qubits**: {result['circuit_qubits']}
- **Circuit Depth**: {result['circuit_depth']}
- **Gate Types**: {gate_summary}
"""
        
        report += f"""

*Generated on {time.strftime('%Y-%m-%d %H:%M:%S')} using Cirq {cirq.__version__}*
"""
        
        return report
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types and other non-serializable objects to JSON-compatible types"""
        if isinstance(obj, dict):
            return {str(k): self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_json_serializable(item) for item in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def save_results(self, filename: str = "quantum_ecc_cirq_results.json"):
        """Save experimental results to JSON file"""
        output_data = {
            'framework': 'Cirq',
            'hardware_info': self.hardware_info,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': self.results,
            'analysis': self.analyze_success_rates()
        }
        
        # Convert to JSON-serializable format
        serializable_data = self._convert_to_json_serializable(output_data)
        
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"Results saved to {filename}")


def main():
    """Main experimental pipeline"""
    print("🚀 Starting Quantum ECC Attack Simulation with Cirq")
    print("=" * 60)
    
    # Initialize experiment framework
    experiment = QuantumECCExperiment()
    
    # Run parameter sweep
    results = experiment.run_parameter_sweep()
    
    # Analyze and visualize results
    print("\n" + "="*60)
    print("ANALYSIS AND VISUALIZATION")
    print("="*60)
    
    analysis = experiment.analyze_success_rates()
    print(f"Experimental Analysis Complete:")
    print(f"- Total experiments: {analysis['total_experiments']}")
    print(f"- Quantum success rate: {analysis['quantum_success_rate']:.2%}")
    print(f"- Average quantum time: {analysis['average_quantum_time']:.4f}s")
    
    # Generate plots
    experiment.plot_results()
    
    # Generate and save report
    report = experiment.generate_report()
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print(report)
    
    # Save results
    experiment.save_results()
    
    print("\n🎉 Quantum ECC Attack Simulation Complete!")
    print("Check the generated plots and saved JSON file for detailed results.")


if __name__ == "__main__":
    main()