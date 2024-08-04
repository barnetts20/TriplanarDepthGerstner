struct WaveOutput{
    float3 Displacement;
    float3 Tangent;
    float3 Binormal;
};

struct CompositeWaveOutput{
    float3 Displacement;
    float3 Normal;
    float CrestMask;
};

struct GerstnerFunctions {
    float3 POSITION;
    float SCALE;
    float DEPTH;
    float ADJ_DEPTH;
    float TIME;
    float3 VERTEX_NORMAL;
    float MIN_DEPTH_CUTOFF;
    float MAX_DEPTH_CUTOFF;
    float MIN_DEPTH_FREQUENCY;
    float DEPTH_WAVE_SHARPENING;
    float MIN_AMPLITUDE;
    float2 DIRECTION_BOUNDS_X;
    float2 DIRECTION_BOUNDS_Y;
    float2 STEEPNESS_BOUNDS;
    float2 WAVELENGTH_BOUNDS;

    float WAVE_FLATNESS;
    int NUM_WAVES;
    int WAVES_PER_OCTAVE;
    float OCTAVE_SCALE;

    Texture2D NOISE_TEX;
    SamplerState NOISE_TEX_SAMPLER;

    WaveOutput GerstnerWave(float4 wave, float3 position){
        WaveOutput waveOut;
        float depthAdjustedWavelength = wave.w * ADJ_DEPTH * ADJ_DEPTH;
        float k1 = 2 * PI / depthAdjustedWavelength;
        float k2 = 2 * PI / wave.w;
        float c = sqrt(9.8 / k2) / k1;
        float2 d = normalize(wave.xy);
        float f = k1 * (dot(d, position.xy) - c * TIME);
        float a = wave.z * saturate(DEPTH + MIN_AMPLITUDE) / k1;

        float cosf = cos(f);
        float sinf = -sin(f);

        waveOut.Displacement = float3(
            d.x * (a * cosf),
            d.y * (a * cosf),
            a * sinf
        );

        waveOut.Tangent = float3(
            1 - d.x * d.x * wave.z * sinf,
            -d.y * d.y * wave.z * sinf,
            d.x * wave.z * cosf
        );

        waveOut.Binormal = float3(
            -d.x * d.y * wave.z * sinf,
            1 - d.y * d.y * wave.z * sinf,
            d.y * wave.z * cosf
        );

        return waveOut;
    }

    CompositeWaveOutput CompositeGerstnerWaves(float3 position){
        CompositeWaveOutput compositeWaveOut;
        float3 accumulatedDisplacement = float3(0,0,0);
        float3 accumulatedTangent = float3(0,0,0);
        float3 accumulatedBinormal = float3(0,0,0);
        float accumulatedJacobian = 0.0;
        float currentOctaveMultiplier = 1;
        float accumulationCoefficient = WAVE_FLATNESS;
        for(int i = 0; i < NUM_WAVES; i++){
            if(i > 0 && i % WAVES_PER_OCTAVE == 0){
                currentOctaveMultiplier *= OCTAVE_SCALE;
            }
            float samplePos = ((float)i/NUM_WAVES);
            float4 waveRand = Texture2DSample(NOISE_TEX, NOISE_TEX_SAMPLER, float2(samplePos,samplePos));
            float4 wave = float4(
                lerp(DIRECTION_BOUNDS_X.x, DIRECTION_BOUNDS_X.y, waveRand.x),
                lerp(DIRECTION_BOUNDS_Y.x, DIRECTION_BOUNDS_Y.y, waveRand.y),
                lerp(STEEPNESS_BOUNDS.x, STEEPNESS_BOUNDS.y, waveRand.z),
                lerp(WAVELENGTH_BOUNDS.x, WAVELENGTH_BOUNDS.y, waveRand.w) * SCALE * currentOctaveMultiplier
            );
            accumulationCoefficient += wave.z * currentOctaveMultiplier;
            WaveOutput currentWaveOutput = GerstnerWave(wave, position);
            accumulatedDisplacement += currentWaveOutput.Displacement / accumulationCoefficient;
            accumulatedTangent += currentWaveOutput.Tangent / accumulationCoefficient;
            accumulatedBinormal += currentWaveOutput.Binormal / accumulationCoefficient;
        }

        compositeWaveOut.Displacement = accumulatedDisplacement * sign(VERTEX_NORMAL);
        compositeWaveOut.Normal = -normalize(cross(accumulatedBinormal, accumulatedTangent) + VERTEX_NORMAL);
        compositeWaveOut.CrestMask = smoothstep(.0,1,saturate(1-length(compositeWaveOut.Displacement.xy/((WAVELENGTH_BOUNDS.x + WAVELENGTH_BOUNDS.y)/2)))) * smoothstep(0, 1, accumulatedDisplacement.z/DEPTH);
        return compositeWaveOut;
    }
};

GerstnerFunctions gf;
//Base parameters
gf.POSITION = Position;
gf.SCALE = Scale;
gf.DEPTH = Depth;
gf.ADJ_DEPTH = (.8 + Depth * .2) * 1.0;
gf.TIME = Time;

//Depth interaction parameters
gf.MIN_DEPTH_CUTOFF = MinDepthCutoff;
gf.MAX_DEPTH_CUTOFF = MaxDepthCutoff;
gf.MIN_DEPTH_FREQUENCY = MinDepthFrequency;
gf.DEPTH_WAVE_SHARPENING = DepthWaveSharpening;
gf.MIN_AMPLITUDE = .025;
//Randomization bounds
gf.DIRECTION_BOUNDS_X = float2(-1,1);
gf.DIRECTION_BOUNDS_Y = float2(-1,1);
gf.STEEPNESS_BOUNDS = float2(.2,.6);
gf.WAVELENGTH_BOUNDS = float2(.5,1.5);
gf.WAVE_FLATNESS = 2;
gf.NUM_WAVES = 100;
gf.WAVES_PER_OCTAVE = 15;
gf.OCTAVE_SCALE = .5;
gf.VERTEX_NORMAL = VertexNormal;
gf.NOISE_TEX = NoiseTex;
gf.NOISE_TEX_SAMPLER = NoiseTexSampler;

CompositeWaveOutput wx = gf.CompositeGerstnerWaves(float3(Position.y, Position.z, Position.x));
CompositeWaveOutput wy = gf.CompositeGerstnerWaves(float3(Position.x, Position.z, Position.y));
CompositeWaveOutput wz = gf.CompositeGerstnerWaves(float3(Position.x, Position.y, Position.z));

float3 vn = pow(abs(VertexNormal), 16);
vn = vn / (vn.x + vn.y + vn.z);

float3 displacement = wx.Displacement * vn.x + wy.Displacement * vn.y + wz.Displacement * vn.z;
Normal = wx.Normal * vn.x + wy.Normal * vn.y + wz.Normal * vn.z;
CrestMask = wx.CrestMask * vn.x + wy.CrestMask * vn.y + wz.CrestMask * vn.z;

return displacement;
