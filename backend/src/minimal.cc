// Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"

namespace triton { namespace backend { namespace minimal {

//
// Minimal backend that demonstrates the TRITONBACKEND API. This
// backend works for any model that has 1 input called "IN0" with
// INT32 datatype and shape [ 4 ] and 1 output called "OUT0" with
// INT32 datatype and shape [ 4 ]. The backend supports both batching
// and non-batching models.
//
// For each batch of requests, the backend returns the input tensor
// value in the output tensor.
//

/////////////

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model. ModelState is derived from BackendModel class
// provided in the backend utilities that provides many common
// functions.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

 private:
  ModelState(TRITONBACKEND_Model* triton_model) : BackendModel(triton_model) {}
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

extern "C" {

// Triton calls TRITONBACKEND_ModelInitialize when a model is loaded
// to allow the backend to create any state associated with the model,
// and to also examine the model configuration to determine if the
// configuration is suitable for the backend. Any errors reported by
// this function will prevent the model from loading.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model. If anything goes wrong with initialization
  // of the model state then an error is returned and Triton will fail
  // to load the model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelFinalize when a model is no longer
// needed. The backend should cleanup any state associated with the
// model. This function will not be called until all model instances
// of the model have been finalized.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);
  delete model_state;

  return nullptr;  // success
}

}  // extern "C"

/////////////

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each
// TRITONBACKEND_ModelInstance. ModelInstanceState is derived from
// BackendModelInstance class provided in the backend utilities that
// provides many common functions.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState() = default;

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance)
      : BackendModelInstance(model_state, triton_model_instance),
        model_state_(model_state)
  {
  }

  ModelState* model_state_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

extern "C" {

// Triton calls TRITONBACKEND_ModelInstanceInitialize when a model
// instance is created to allow the backend to initialize any state
// associated with the instance.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelInstanceFinalize when a model
// instance is no longer needed. The backend should cleanup any state
// associated with the model instance.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);
  delete instance_state;

  return nullptr;  // success
}

}  // extern "C"

/////////////

extern "C" {

// When Triton calls TRITONBACKEND_ModelInstanceExecute it is required
// that a backend create a response for each request in the batch. A
// response may be the output tensors required for that request or may
// be an error that is returned in the response.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
    ModelInstanceState* instance_state;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
        instance, reinterpret_cast<void**>(&instance_state)));
    ModelState* model_state = instance_state->StateForModel();

    std::vector<TRITONBACKEND_Response*> responses;
    responses.reserve(request_count);
    for (uint32_t r = 0; r < request_count; ++r) {
        TRITONBACKEND_Response* response;
        RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, requests[r]));
        responses.push_back(response);
    }

    // Initialize the BackendInputCollector
    BackendInputCollector collector(
        requests, request_count, &responses, model_state->TritonMemoryManager(),
        false /* pinned_enabled */, nullptr /* stream*/);

    // Collect the input tensor
    std::vector<const void*> input_buffers;
    std::vector<size_t> input_buffer_byte_sizes;
    std::vector<TRITONSERVER_MemoryType> input_buffer_memory_types;
    std::vector<int64_t> input_buffer_memory_type_ids;

    for (uint32_t r = 0; r < request_count; ++r) {
        TRITONBACKEND_Input* input;
        RETURN_IF_ERROR(TRITONBACKEND_RequestInputByIndex(requests[r], 0, &input));

        const void* input_buffer;
        size_t input_buffer_byte_size;
        TRITONSERVER_MemoryType input_buffer_memory_type;
        int64_t input_buffer_memory_type_id;
        RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
            input, 0, &input_buffer, &input_buffer_byte_size,
            &input_buffer_memory_type, &input_buffer_memory_type_id));

        input_buffers.push_back(input_buffer);
        input_buffer_byte_sizes.push_back(input_buffer_byte_size);
        input_buffer_memory_types.push_back(input_buffer_memory_type);
        input_buffer_memory_type_ids.push_back(input_buffer_memory_type_id);
    }

    // Process the collected input tensors
    for (size_t i = 0; i < input_buffers.size(); ++i) {
        std::cout << input_buffers[i] << std::endl;
        const double* input_data = reinterpret_cast<const double*>(input_buffers[i]);
        size_t input_size = input_buffer_byte_sizes[i] / sizeof(double);

        // Here you can implement your logic to process the input_data
        // For demonstration, let's just print the first few elements
        std::cout << "Received input (first 5 elements): ";
        for (size_t j = 0; j < std::min(size_t(9), input_size); ++j) {
            std::cout << input_data[j] << " ";
        }
        std::cout << std::endl;
    }

    // Prepare the output tensor
    for (uint32_t r = 0; r < request_count; ++r) {
        TRITONBACKEND_Output* output;
        // Define the shape of the output tensor
        int64_t output_shape[1] = {static_cast<int64_t>(input_buffer_byte_sizes[r] / sizeof(uint64_t))};

        RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
            responses[r], &output, "OUTPUT", TRITONSERVER_TYPE_UINT64, output_shape, 1));

        // Set the output buffer
        void* output_buffer;
        RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(
            output, &output_buffer, input_buffer_byte_sizes[r], &input_buffer_memory_types[r],
            &input_buffer_memory_type_ids[r]));

        // Copy the input data to the output buffer
        std::memcpy(output_buffer, input_buffers[r], input_buffer_byte_sizes[r]);
    }

    // Send the responses
    for (uint32_t r = 0; r < request_count; ++r) {
        if (responses[r] != nullptr) {
            TRITONBACKEND_ResponseSend(
                responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr);
        }
    }

    return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::minimal
