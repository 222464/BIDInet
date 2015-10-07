constant sampler_t normalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP |
	CLK_FILTER_NEAREST;

constant sampler_t normalizedClampedToEdgeNearestSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;

constant sampler_t unnormalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP |
	CLK_FILTER_NEAREST;

constant sampler_t defaultNormalizedSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;

constant sampler_t defaultUnnormalizedSampler = CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;

constant float reluLeak = 0.01f;

float randFloat(uint2* state) {
	const float invMaxInt = 1.0f / 4294967296.0f;
	uint x = (*state).x * 17 + (*state).y * 13123;
	(*state).x = (x << 13) ^ x;
	(*state).y ^= (x << 7);

	uint tmp = x * (x * x * 15731 + 74323) + 871483;

	return convert_float(tmp) * invMaxInt;
}

float randNormal(uint2* state) {
	float u1 = randFloat(state);
	float u2 = randFloat(state);

	return sqrt(-2.0f * log(u1)) * cos(6.28318f * u2);
}

float sigmoid(float x) {
	return 1.0f / (1.0f + exp(-x));
}

float relu(float x) {
	//return (x > 0.0f && x < 1.0f) ? x : (1.0f + reluLeak * (x - 1.0f));
	return x > 0.0f ? x : reluLeak * x;
}

float relud(float x) {
	//return (x > 0.0f && x < 1.0f) ? 1.0f : reluLeak;
	return x > 0.0f ? 1.0f : reluLeak;
}

void kernel initializeConnections(write_only image3d_t connections,
	int size, uint2 seed, float minWeight, float maxWeight)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 29 + 12, get_global_id(1) * 16 + 23) * 36;

	int2 position = (int2)(get_global_id(0), get_global_id(1));

	for (int ci = 0; ci < size; ci++) {
		int4 connectionPosition = (int4)(position.x, position.y, ci, 0);

		float weight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;

		write_imagef(connections, connectionPosition, (float4)(weight, 0.0f, 0.0f, 0.0f));
	}
}

void kernel ffActivate(read_only image2d_t inputs, read_only image2d_t ffStatesPrev,
	read_only image3d_t ffConnections, read_only image3d_t recConnections,
	write_only image2d_t ffActivations,
	int2 layerSize, int2 inputsSize,
	int ffRadius, int recRadius,
	float2 layerToInputsScalar)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	int2 ffCenter = (int2)(position.x * layerToInputsScalar.x + 0.5f, position.y * layerToInputsScalar.y + 0.5f);

	float activation = 0.0f;

	int ci;
	
	ci = 0;

	for (int dx = -ffRadius; dx <= ffRadius; dx++)
		for (int dy = -ffRadius; dy <= ffRadius; dy++) {
			int2 ffPosition = ffCenter + (int2)(dx, dy);

			if (ffPosition.x >= 0 && ffPosition.x < inputsSize.x && ffPosition.y >= 0 && ffPosition.y < inputsSize.y) {
				float connection = read_imagef(ffConnections, (int4)(position.x, position.y, ci, 0)).x;

				float input = read_imagef(inputs, ffPosition).x;

				float delta = input - connection;

				activation -= delta * delta;
			}

			ci++;
		}

	// Bias
	float biasConnection = read_imagef(ffConnections, (int4)(position.x, position.y, ci, 0)).x;

	activation += biasConnection;

	ci = 0;

	for (int dx = -recRadius; dx <= recRadius; dx++)
		for (int dy = -recRadius; dy <= recRadius; dy++) {
			int2 recPosition = position + (int2)(dx, dy);

			if (recPosition.x >= 0 && recPosition.x < layerSize.x && recPosition.y >= 0 && recPosition.y < layerSize.y) {
				float connection = read_imagef(recConnections, (int4)(position.x, position.y, ci, 0)).x;

				float input = read_imagef(ffStatesPrev, recPosition).x;

				float delta = input - connection;

				activation -= delta * delta;
			}

			ci++;
		}

	// Write result
	write_imagef(ffActivations, position, (float4)(activation, 0.0f, 0.0f, 0.0f));
}

void kernel ffInhibit(read_only image2d_t ffActivations,
	read_only image3d_t lConnections,
	write_only image2d_t ffStates,
	int2 layerSize,
	int lRadius)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float thisActivation = read_imagef(ffActivations, position).x;

	float inhibition = 0.0f;

	int ci;

	ci = 0;

	for (int dx = -lRadius; dx <= lRadius; dx++)
		for (int dy = -lRadius; dy <= lRadius; dy++) {
			int2 lPosition = position + (int2)(dx, dy);

			if (lPosition.x != position.x || lPosition.y != position.y) {
				if (lPosition.x >= 0 && lPosition.x < layerSize.x && lPosition.y >= 0 && lPosition.y < layerSize.y) {
					float connection = read_imagef(lConnections, (int4)(position.x, position.y, ci, 0)).x;

					float activation = read_imagef(ffActivations, lPosition).x;

					inhibition += activation > thisActivation ? connection : 0.0f;
				}
			}

			ci++;
		}

	float state = inhibition > 1.0f ? 0.0f : 1.0f;

	// Write result
	write_imagef(ffStates, position, (float4)(state, 0.0f, 0.0f, 0.0f));
}

void kernel fbActivate(read_only image2d_t inputs, read_only image2d_t ffStates,
	read_only image3d_t fbConnections, read_only image3d_t predConnections,
	write_only image2d_t fbStates, write_only image2d_t explorations,
	int2 inputsSize, int2 layerSize,
	int fbRadius, int predRadius,
	float2 layerToInputsScalar,
	float breakChance,
	uint2 seed)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 23 + 4, get_global_id(1) * 7 + 56) * 4;

	int2 position = (int2)(get_global_id(0), get_global_id(1));

	int2 fbCenter = (int2)(position.x * layerToInputsScalar.x + 0.5f, position.y * layerToInputsScalar.y + 0.5f);

	float activation = 0.0f;

	int ci;

	ci = 0;

	for (int dx = -fbRadius; dx <= fbRadius; dx++)
		for (int dy = -fbRadius; dy <= fbRadius; dy++) {
			int2 fbPosition = fbCenter + (int2)(dx, dy);

			if (fbPosition.x >= 0 && fbPosition.x < inputsSize.x && fbPosition.y >= 0 && fbPosition.y < inputsSize.y) {
				float connection = read_imagef(fbConnections, (int4)(position.x, position.y, ci, 0)).x;

				float input = read_imagef(inputs, fbPosition).x;

				activation += connection * input;
			}

			ci++;
		}

	float biasConnection = read_imagef(fbConnections, (int4)(position.x, position.y, ci, 0)).x;

	activation += biasConnection;

	ci = 0;

	for (int dx = -predRadius; dx <= predRadius; dx++)
		for (int dy = -predRadius; dy <= predRadius; dy++) {
			int2 predPosition = position + (int2)(dx, dy);

			if (predPosition.x >= 0 && predPosition.x < layerSize.x && predPosition.y >= 0 && predPosition.y < layerSize.y) {
				float connection = read_imagef(predConnections, (int4)(position.x, position.y, ci, 0)).x;

				float input = read_imagef(ffStates, predPosition).x;

				activation += connection * input;
			}

			ci++;
		}

	float state = sigmoid(activation);

	float stateExp = state;
	
	if (randFloat(&seedValue) < breakChance)
		stateExp = randFloat(&seedValue);

	// Write result
	write_imagef(fbStates, position, (float4)(state, 0.0f, 0.0f, 0.0f));
	write_imagef(explorations, position, (float4)(stateExp, 0.0f, 0.0f, 0.0f));
}

void kernel fbActivateFirst(read_only image2d_t inputs,
	read_only image3d_t fbConnections, read_only image3d_t predConnections,
	write_only image2d_t fbStates, write_only image2d_t explorations,
	int2 inputsSize, int2 layerSize,
	int fbRadius, int predRadius,
	float2 layerToInputsScalar,
	float breakChance,
	uint2 seed)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 23 + 4, get_global_id(1) * 7 + 56) * 4;

	int2 position = (int2)(get_global_id(0), get_global_id(1));

	int2 fbCenter = (int2)(position.x * layerToInputsScalar.x + 0.5f, position.y * layerToInputsScalar.y + 0.5f);

	float activation = 0.0f;

	int ci;

	ci = 0;

	for (int dx = -fbRadius; dx <= fbRadius; dx++)
		for (int dy = -fbRadius; dy <= fbRadius; dy++) {
			int2 fbPosition = fbCenter + (int2)(dx, dy);

			if (fbPosition.x >= 0 && fbPosition.x < inputsSize.x && fbPosition.y >= 0 && fbPosition.y < inputsSize.y) {
				float connection = read_imagef(fbConnections, (int4)(position.x, position.y, ci, 0)).x;

				float input = read_imagef(inputs, fbPosition).x;

				activation += connection * input;
			}

			ci++;
		}

	float biasConnection = read_imagef(fbConnections, (int4)(position.x, position.y, ci, 0)).x;

	//activation += biasConnection;

	float state = sigmoid(activation);

	float stateExp = state;

	if (randFloat(&seedValue) < breakChance)
		stateExp = randFloat(&seedValue);

	// Write result
	write_imagef(fbStates, position, (float4)(state, 0.0f, 0.0f, 0.0f));
	write_imagef(explorations, position, (float4)(stateExp, 0.0f, 0.0f, 0.0f));
}

void kernel ffReconstruct(read_only image2d_t ffStates, read_only image3d_t ffConnections, write_only image2d_t reconstruction,
	int ffRadius, int2 ffReverseRadius,
	int2 inputSize, int2 layerSize,
	float2 layerToInputsScalar, float2 inputsToLayerScalar)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	int2 layerCenterPosition = (int2)(position.x * inputsToLayerScalar.x + 0.5f, position.y * inputsToLayerScalar.y + 0.5f);

	float recon = 0.0f;
	float div = 0.0f;

	for (int dx = -ffReverseRadius.x; dx <= ffReverseRadius.x; dx++)
		for (int dy = -ffReverseRadius.y; dy <= ffReverseRadius.y; dy++) {
			int2 layerPosition = layerCenterPosition + (int2)(dx, dy);

			if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
				// Next layer node's receptive field
				int2 fieldCenter = (int2)(layerPosition.x * layerToInputsScalar.x + 0.5f, layerPosition.y * layerToInputsScalar.y + 0.5f);

				int2 fieldLowerBounds = fieldCenter - (int2)(ffRadius);
				int2 fieldUpperBounds = fieldCenter + (int2)(ffRadius);

				// Check for containment
				if (position.x >= fieldLowerBounds.x && position.x <= fieldUpperBounds.x && position.y >= fieldLowerBounds.y && position.y <= fieldUpperBounds.y) {
					int2 rd = position - fieldLowerBounds;

					int ci = rd.y + rd.x * (ffRadius * 2 + 1);

					float state = read_imagef(ffStates, layerPosition).x;
					float connection = read_imagef(ffConnections, (int4)(layerPosition.x, layerPosition.y, ci, 0)).x;

					recon += state * connection;
					div += state;
				}
			}
		}

	write_imagef(reconstruction, position, (float4)(recon / fmax(1.0f, div), 0.0f, 0.0f, 0.0f));
}

void kernel recReconstruct(read_only image2d_t ffStates, read_only image3d_t recConnections, write_only image2d_t reconstruction,
	int recRadius,
	int2 layerSize)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float recon = 0.0f;
	float div = 0.0f;

	for (int dx = -recRadius; dx <= recRadius; dx++)
		for (int dy = -recRadius; dy <= recRadius; dy++) {
			int2 inputPosition = position + (int2)(dx, dy);

			if (inputPosition.x >= 0 && inputPosition.x < layerSize.x && inputPosition.y >= 0 && inputPosition.y < layerSize.y) {
				int ci = (recRadius - dy) + (recRadius - dx) * (recRadius * 2 + 1);

				float state = read_imagef(ffStates, inputPosition).x;
				float connection = read_imagef(recConnections, (int4)(inputPosition.x, inputPosition.y, ci, 0)).x;

				recon += state * connection;
				div += state;
			}
		}

	write_imagef(reconstruction, position, (float4)(recon / fmax(1.0f, div), 0.0f, 0.0f, 0.0f));
}

void kernel ffConnectionUpdate(read_only image2d_t inputs,
	read_only image3d_t ffConnectionsPrev, write_only image3d_t ffConnections,
	read_only image2d_t ffReconstruction,
	read_only image2d_t ffStates,
	int2 layerSize, int2 inputsSize,
	int ffRadius,
	float2 layerToInputsScalar,
	float ffAlpha, float ffGamma, float sparsity)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	int2 ffCenter = (int2)(position.x * layerToInputsScalar.x + 0.5f, position.y * layerToInputsScalar.y + 0.5f);

	float ffState = read_imagef(ffStates, position).x;

	int ci;

	ci = 0;

	for (int dx = -ffRadius; dx <= ffRadius; dx++)
		for (int dy = -ffRadius; dy <= ffRadius; dy++) {
			int2 ffPosition = ffCenter + (int2)(dx, dy);

			if (ffPosition.x >= 0 && ffPosition.x < inputsSize.x && ffPosition.y >= 0 && ffPosition.y < inputsSize.y) {
				float connectionPrev = read_imagef(ffConnectionsPrev, (int4)(position.x, position.y, ci, 0)).x;

				float input = read_imagef(inputs, ffPosition).x;
				float recon = read_imagef(ffReconstruction, ffPosition).x;

				float connection = connectionPrev + ffAlpha * ffState * (input - recon);

				write_imagef(ffConnections, (int4)(position.x, position.y, ci, 0), (float4)(connection, 0.0f, 0.0f, 0.0f));
			}

			ci++;
		}

	// Bias
	float biasConnectionPrev = read_imagef(ffConnectionsPrev, (int4)(position.x, position.y, ci, 0)).x;

	float biasConnection = biasConnectionPrev + ffGamma * (sparsity - ffState);

	write_imagef(ffConnections, (int4)(position.x, position.y, ci, 0), (float4)(biasConnection, 0.0f, 0.0f, 0.0f));
}

void kernel recConnectionUpdate(read_only image2d_t ffStatesPrev,
	read_only image3d_t recConnectionsPrev, write_only image3d_t recConnections,
	read_only image2d_t recReconstruction,
	read_only image2d_t ffStates,
	int2 layerSize,
	int recRadius,
	float ffAlpha)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float ffState = read_imagef(ffStates, position).x;

	int ci;

	ci = 0;

	for (int dx = -recRadius; dx <= recRadius; dx++)
		for (int dy = -recRadius; dy <= recRadius; dy++) {
			int2 recPosition = position + (int2)(dx, dy);

			if (recPosition.x >= 0 && recPosition.x < layerSize.x && recPosition.y >= 0 && recPosition.y < layerSize.y) {
				float connectionPrev = read_imagef(recConnectionsPrev, (int4)(position.x, position.y, ci, 0)).x;

				float input = read_imagef(ffStatesPrev, recPosition).x;
				float recon = read_imagef(recReconstruction, recPosition).x;

				float connection = connectionPrev + ffAlpha * ffState * (input - recon);

				write_imagef(recConnections, (int4)(position.x, position.y, ci, 0), (float4)(connection, 0.0f, 0.0f, 0.0f));
			}

			ci++;
		}
}

void kernel lConnectionUpdate(read_only image2d_t ffActivations, read_only image2d_t ffStates,
	read_only image3d_t lConnectionsPrev, write_only image3d_t lConnections,
	int2 layerSize,
	int lRadius,
	float ffBeta, float sparsitySquared)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float ffState = read_imagef(ffStates, position).x;
	float ffActivation = read_imagef(ffActivations, position).x;

	int ci;

	ci = 0;

	for (int dx = -lRadius; dx <= lRadius; dx++)
		for (int dy = -lRadius; dy <= lRadius; dy++) {
			int2 lPosition = position + (int2)(dx, dy);

			if (lPosition.x >= 0 && lPosition.x < layerSize.x && lPosition.y >= 0 && lPosition.y < layerSize.y) {
				float connectionPrev = read_imagef(lConnectionsPrev, (int4)(position.x, position.y, ci, 0)).x;

				float input = read_imagef(ffActivations, lPosition).x;

				float connection = fmax(0.0f, connectionPrev + ffBeta * (ffState * (input < ffActivation ? 1.0f : 0.0f) - sparsitySquared));

				write_imagef(lConnections, (int4)(position.x, position.y, ci, 0), (float4)(connection, 0.0f, 0.0f, 0.0f));
			}

			ci++;
		}
}

void kernel fbConnectionUpdate(read_only image2d_t inputsPrev, read_only image2d_t inputs,
	read_only image3d_t fbConnectionsPrev, write_only image3d_t fbConnections,
	read_only image2d_t fbStatesPrev, read_only image2d_t fbStates, read_only image2d_t ffStates,
	read_only image2d_t explorations,
	int2 inputsSize,
	int fbRadius,
	float2 layerToInputsScalar,
	float fbPredAlpha, float fbRLAlpha, float fbLambdaGamma,
	float rlError)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	int2 fbCenter = (int2)(position.x * layerToInputsScalar.x + 0.5f, position.y * layerToInputsScalar.y + 0.5f);

	float fbState = read_imagef(fbStates, position).x;
	float fbStatePrev = read_imagef(fbStatesPrev, position).x;
	float ffState = read_imagef(ffStates, position).x;

	float delta = read_imagef(explorations, position).x - fbState;

	float predError = ffState - fbStatePrev;

	int ci;

	ci = 0;

	for (int dx = -fbRadius; dx <= fbRadius; dx++)
		for (int dy = -fbRadius; dy <= fbRadius; dy++) {
			int2 fbPosition = fbCenter + (int2)(dx, dy);

			if (fbPosition.x >= 0 && fbPosition.x < inputsSize.x && fbPosition.y >= 0 && fbPosition.y < inputsSize.y) {
				float2 connectionPrev = read_imagef(fbConnectionsPrev, (int4)(position.x, position.y, ci, 0)).xy;

				float input = read_imagef(inputs, fbPosition).x;
				float inputPrev = read_imagef(inputsPrev, fbPosition).x;

				float2 connection = (float2)(connectionPrev.x + fbRLAlpha * rlError * connectionPrev.y + fbPredAlpha * predError * inputPrev, connectionPrev.y * fbLambdaGamma + delta * input);
				
				write_imagef(fbConnections, (int4)(position.x, position.y, ci, 0), (float4)(connection.x, connection.y, 0.0f, 0.0f));
			}

			ci++;
		}

	float2 biasConnectionPrev = read_imagef(fbConnectionsPrev, (int4)(position.x, position.y, ci, 0)).xy;

	float2 biasConnection = (float2)(biasConnectionPrev.x + fbRLAlpha * rlError * biasConnectionPrev.y + fbPredAlpha * predError, biasConnectionPrev.y * fbLambdaGamma + delta);

	write_imagef(fbConnections, (int4)(position.x, position.y, ci, 0), (float4)(biasConnection.x, biasConnection.y, 0.0f, 0.0f));
}

void kernel predConnectionUpdate(read_only image3d_t predConnectionsPrev, write_only image3d_t predConnections,
	read_only image2d_t fbStatesPrev, read_only image2d_t fbStates, read_only image2d_t ffStatesPrev, read_only image2d_t ffStates,
	read_only image2d_t explorations,
	int2 layerSize,
	int predRadius,
	float fbPredAlpha, float fbRLAlpha, float fbLambdaGamma,
	float rlError)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float fbState = read_imagef(fbStates, position).x;
	float fbStatePrev = read_imagef(fbStatesPrev, position).x;
	float ffState = read_imagef(ffStates, position).x;

	float delta = read_imagef(explorations, position).x - fbState;

	float predError = ffState - fbStatePrev;

	int ci;

	ci = 0;

	for (int dx = -predRadius; dx <= predRadius; dx++)
		for (int dy = -predRadius; dy <= predRadius; dy++) {
			int2 predPosition = position + (int2)(dx, dy);

			if (predPosition.x >= 0 && predPosition.x < layerSize.x && predPosition.y >= 0 && predPosition.y < layerSize.y) {
				float2 connectionPrev = read_imagef(predConnectionsPrev, (int4)(position.x, position.y, ci, 0)).xy;

				float input = read_imagef(ffStates, predPosition).x;
				float inputPrev = read_imagef(ffStatesPrev, predPosition).x;

				float2 connection = (float2)(connectionPrev.x + fbRLAlpha * rlError * connectionPrev.y + fbPredAlpha * predError * inputPrev, connectionPrev.y * fbLambdaGamma + delta * input);

				write_imagef(predConnections, (int4)(position.x, position.y, ci, 0), (float4)(connection.x, connection.y, 0.0f, 0.0f));
			}

			ci++;
		}

	float2 biasConnectionPrev = read_imagef(predConnectionsPrev, (int4)(position.x, position.y, ci, 0)).xy;

	float2 biasConnection = (float2)(biasConnectionPrev.x + fbRLAlpha * rlError * biasConnectionPrev.y + fbPredAlpha * predError, biasConnectionPrev.y * fbLambdaGamma + delta);

	write_imagef(predConnections, (int4)(position.x, position.y, ci, 0), (float4)(biasConnection.x, biasConnection.y, 0.0f, 0.0f));
}