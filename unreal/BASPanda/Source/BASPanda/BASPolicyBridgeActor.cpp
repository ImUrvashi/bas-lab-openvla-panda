#include "BASPolicyBridgeActor.h"

#include "BASPolicyCameraActor.h"
#include "PandaRobotActor.h"
#include "Dom/JsonObject.h"
#include "HttpModule.h"
#include "Kismet/GameplayStatics.h"
#include "Misc/Base64.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

ABASPolicyBridgeActor::ABASPolicyBridgeActor()
{
    PrimaryActorTick.bCanEverTick = false;
}

void ABASPolicyBridgeActor::BeginPlay()
{
    Super::BeginPlay();
    if (bSendOnBeginPlay)
    {
        SendTestObservation();
    }
}

void ABASPolicyBridgeActor::SendTestObservation()
{
    ABASPolicyCameraActor* Camera = Cast<ABASPolicyCameraActor>(
        UGameplayStatics::GetActorOfClass(this, ABASPolicyCameraActor::StaticClass()));
    if (!Camera)
    {
        LastResult = TEXT("No BASPolicyCameraActor in this level");
        UE_LOG(LogTemp, Error, TEXT("BAS policy test: %s"), *LastResult);
        return;
    }

    TArray<uint8> RGB;
    FString Error;
    if (!Camera->CaptureRGB(RGB, Error))
    {
        LastResult = Error;
        UE_LOG(LogTemp, Error, TEXT("BAS policy test: %s"), *LastResult);
        return;
    }

    TSharedRef<FJsonObject> Payload = MakeShared<FJsonObject>();
    Payload->SetStringField(TEXT("instruction"), Instruction);
    Payload->SetStringField(TEXT("image_b64"), FBase64::Encode(RGB.GetData(), RGB.Num()));
    TArray<TSharedPtr<FJsonValue>> Shape;
    for (const int32 Dimension : {224, 224, 3})
    {
        Shape.Add(MakeShared<FJsonValueNumber>(Dimension));
    }
    Payload->SetArrayField(TEXT("shape"), Shape);
    APandaRobotActor* Panda = Cast<APandaRobotActor>(
        UGameplayStatics::GetActorOfClass(this, APandaRobotActor::StaticClass()));
    if (Panda && Panda->ArmJointsRad.Num() == 7)
    {
        TArray<TSharedPtr<FJsonValue>> Joints;
        for (const float Angle : Panda->ArmJointsRad)
        {
            Joints.Add(MakeShared<FJsonValueNumber>(Angle));
        }
        Payload->SetArrayField(TEXT("joints_rad"), Joints);
        Payload->SetNumberField(TEXT("gripper_opening"), FMath::Clamp(Panda->FingerOpeningCm / 4.0f, 0.0f, 1.0f));
    }
    FString JsonBody;
    const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&JsonBody);
    FJsonSerializer::Serialize(Payload, Writer);

    TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Request = FHttpModule::Get().CreateRequest();
    Request->SetURL(PolicyUrl);
    Request->SetVerb(TEXT("POST"));
    Request->SetHeader(TEXT("Content-Type"), TEXT("application/json"));
    Request->SetContentAsString(JsonBody);
    Request->SetTimeout(10.0f);
    Request->OnProcessRequestComplete().BindUObject(this, &ABASPolicyBridgeActor::HandleResponse);
    LastResult = FString::Printf(TEXT("Sending %d RGB bytes"), RGB.Num());
    if (!Request->ProcessRequest())
    {
        LastResult = TEXT("Could not start HTTP request");
        UE_LOG(LogTemp, Error, TEXT("BAS policy test: %s"), *LastResult);
    }
}

void ABASPolicyBridgeActor::HandleResponse(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bSucceeded)
{
    if (!bSucceeded || !Response.IsValid())
    {
        LastResult = TEXT("HTTP connection failed; check that mock_policy_server.py is running on port 8000");
        UE_LOG(LogTemp, Error, TEXT("BAS policy test: %s"), *LastResult);
        return;
    }
    TSharedPtr<FJsonObject> Body;
    const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Response->GetContentAsString());
    if (Response->GetResponseCode() != 200 || !FJsonSerializer::Deserialize(Reader, Body) || !Body.IsValid())
    {
        LastResult = FString::Printf(TEXT("HTTP %d: %s"), Response->GetResponseCode(), *Response->GetContentAsString());
        UE_LOG(LogTemp, Error, TEXT("BAS policy test: %s"), *LastResult);
        return;
    }
    const TArray<TSharedPtr<FJsonValue>>* Action = nullptr;
    if (!Body->TryGetArrayField(TEXT("action"), Action) || !Action || Action->Num() != 7)
    {
        LastResult = TEXT("Policy response has no seven-value action");
        UE_LOG(LogTemp, Error, TEXT("BAS policy test: %s"), *LastResult);
        return;
    }
    if (bApplyJointTargets)
    {
        const TArray<TSharedPtr<FJsonValue>>* Targets = nullptr;
        if (!Body->TryGetArrayField(TEXT("joint_targets"), Targets) || !Targets || Targets->Num() != 8)
        {
            LastResult = TEXT("No eight-value joint_targets in response; no motion applied");
            UE_LOG(LogTemp, Warning, TEXT("BAS policy test: %s"), *LastResult);
            return;
        }
        TArray<float> JointTargets;
        for (int32 Index = 0; Index < 8; ++Index)
        {
            if (!(*Targets)[Index].IsValid() || (*Targets)[Index]->Type != EJson::Number ||
                !FMath::IsFinite((*Targets)[Index]->AsNumber()))
            {
                LastResult = TEXT("Invalid joint_targets; no motion applied");
                UE_LOG(LogTemp, Warning, TEXT("BAS policy test: %s"), *LastResult);
                return;
            }
            if (Index < 7)
            {
                JointTargets.Add(static_cast<float>((*Targets)[Index]->AsNumber()));
            }
        }
        APandaRobotActor* Panda = Cast<APandaRobotActor>(
            UGameplayStatics::GetActorOfClass(this, APandaRobotActor::StaticClass()));
        if (!Panda || !Panda->CommandJointTargets(JointTargets, (*Targets)[7]->AsNumber()))
        {
            LastResult = TEXT("Panda could not accept joint targets; run Simulate with motion test disabled");
            UE_LOG(LogTemp, Warning, TEXT("BAS policy test: %s"), *LastResult);
            return;
        }
        LastResult = TEXT("OK: applied bounded joint target from policy gateway");
    }
    else
    {
        LastResult = FString::Printf(TEXT("OK: received seven values; gripper=%.2f (not applied)"), (*Action)[6]->AsNumber());
    }
    UE_LOG(LogTemp, Display, TEXT("BAS policy test: %s"), *LastResult);
}
