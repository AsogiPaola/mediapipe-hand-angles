// Incluye las cabeceras necesarias
#include "HandMovement.h"
#include "yaml-cpp/yaml.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

AHandMovement::AHandMovement()
{
    PrimaryActorTick.bCanEverTick = true;
}

void AHandMovement::BeginPlay()
{
    Super::BeginPlay();
    
    // Leer el archivo YAML
    FString FilePath = FPaths::ProjectDir() + "angles_data.yaml";
    std::ifstream FileStream(TCHAR_TO_UTF8(*FilePath));
    YAML::Node Data = YAML::Load(FileStream);
    
    for (auto Frame : Data)
    {
        int32 FrameNumber = Frame["frame"].as<int>();
        TArray<FAngleData> Angles;

        for (auto Angle : Frame["angles"])
        {
            FAngleData AngleData;
            AngleData.Finger = FString(Angle["finger"].as<std::string>().c_str());
            AngleData.Angle = Angle["angle"].as<float>();
            Angles.Add(AngleData);
        }

        Frames.Add(FrameNumber, Angles);
    }
}

void AHandMovement::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    if (Frames.Contains(CurrentFrame))
    {
        ApplyAngles(Frames[CurrentFrame]);
        CurrentFrame++;
    }
    else
    {
        CurrentFrame = 0; // Reiniciar si se llega al final
    }
}

void AHandMovement::ApplyAngles(const TArray<FAngleData>& Angles)
{
    for (const auto& AngleData : Angles)
    {
        FRotator Rotation(0.0f, AngleData.Angle, 0.0f);

        if (AngleData.Finger == "thumb")
        {
            Thumb->SetRelativeRotation(Rotation);
        }
        else if (AngleData.Finger == "index")
        {
            Index->SetRelativeRotation(Rotation);
        }
        else if (AngleData.Finger == "middle")
        {
            Middle->SetRelativeRotation(Rotation);
        }
        else if (AngleData.Finger == "ring")
        {
            Ring->SetRelativeRotation(Rotation);
        }
        else if (AngleData.Finger == "pinky")
        {
            Pinky->SetRelativeRotation(Rotation);
        }
    }
}

// HandMovement.h

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "HandMovement.generated.h"

USTRUCT(BlueprintType)
struct FAngleData
{
    GENERATED_BODY()
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString Finger;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Angle;
};

UCLASS()
class YOURPROJECT_API AHandMovement : public AActor
{
    GENERATED_BODY()
    
public:    
    AHandMovement();

protected:
    virtual void BeginPlay() override;

public:    
    virtual void Tick(float DeltaTime) override;

private:
    void ApplyAngles(const TArray<FAngleData>& Angles);

    UPROPERTY(EditAnywhere)
    USceneComponent* Thumb;

    UPROPERTY(EditAnywhere)
    USceneComponent* Index;

    UPROPERTY(EditAnywhere)
    USceneComponent* Middle;

    UPROPERTY(EditAnywhere)
    USceneComponent* Ring;

    UPROPERTY(EditAnywhere)
    USceneComponent* Pinky;

    TMap<int32, TArray<FAngleData>> Frames;
    int32 CurrentFrame = 0;
};
