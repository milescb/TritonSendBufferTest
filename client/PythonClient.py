import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

def main():
    try:
        # Create an HTTP client
        client = httpclient.InferenceServerClient(url='localhost:8000')
        
        # Check if server is live
        if not client.is_server_live():
            print("Server is not live")
            return 1
        
        # Input data as nested list (equivalent to 2D vector in C++)
        input_data = [
            [173748829292910, 2.2, 3.3],
            [4.0, 57.2, 6.0],
            [7.0, 8.0, 9.3455]
        ]
        
        # Convert to numpy array and flatten
        input_data_np = np.array(input_data, dtype=np.float64)
        memory_buffer = np.frombuffer(input_data_np, dtype=np.uint64)
        
        # Get total number of elements
        total_elements = memory_buffer.size
        print(f"Shape: {total_elements}")
        
        # Prepare the input tensor
        inputs = []
        inputs.append(httpclient.InferInput('INPUT', [total_elements], "UINT64"))
        inputs[0].set_data_from_numpy(memory_buffer)
        
        # Prepare the output
        outputs = []
        outputs.append(httpclient.InferRequestedOutput('OUTPUT'))
        
        # Send inference request
        result = client.infer(
            model_name='triton-minimal-backend',
            inputs=inputs,
            outputs=outputs
        )
        
        # Get the result
        output_data = result.as_numpy('OUTPUT')
        output_data = output_data.view(np.float64)
        
        # Reshape output data back to 2D
        num_rows, num_cols = 3, 3
        output_data_2d = output_data.reshape((num_rows, num_cols))
        
        # Print the reshaped output data
        print("Received output:")
        for row in output_data_2d:
            print(" ".join(map(str, row)))
            
        return 0
        
    except InferenceServerException as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    main()