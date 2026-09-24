#pragma once

#include <cstddef>
#include <cstdint>

namespace zonetool::t7::boiii
{
	using vec2_t = float[2];
	using vec3_t = float[3];
	using XString = const char*;
	using AttachmentMask = std::uint64_t;

	namespace scr
	{
		using ScrString_t = std::uint32_t;
	}

	namespace snd
	{
		using SndAliasId = std::uint32_t;
	}

	namespace gfx
	{
		struct GfxImage;
		using GfxImageHandle = GfxImage*;
		union GfxColor
		{
			std::uint32_t packed;
			struct
			{
				std::uint8_t r;
				std::uint8_t g;
				std::uint8_t b;
				std::uint8_t a;
			} rgba;
			std::uint8_t array[4];
		};
	}

	namespace game::db::xasset
	{
		using XModelPtr = void*;
		using ScriptBundlePtr = void*;
	}

	namespace db::xasset
	{
		struct XCam;
		using FxEffectDefHandle = void*;
		using TaxFxSetPtr = void*;
		using SharedWeaponSoundsPtr = void*;
		struct Material;
		using MaterialHandle = Material*;
		using MaterialDeferredHandle = MaterialHandle;
		using ObjectivePtr = void*;
		using FlameTablePtr = void*;
		using BeamDefPtr = void*;
		using FxImpactTablePtr = void*;
		using SoundsImpactTablePtr = void*;
		using TracerDefPtr = void*;
		using LaserDefPtr = void*;
	}

	using AttachmentCosmeticVariantPtr = void*;
	using WeaponAttachmentPtr = void*;
	using WeaponAttachmentUniquePtr = void*;
	using RumbleInfoPtr = void*;

#pragma pack(push, 1)
	struct CamoMaterialChannel
	{
		std::uint16_t replaceFlags;
		std::byte pad_02[6];
		db::xasset::MaterialDeferredHandle camoMaterial;
		float translationX;
		float translationY;
		float scaleX;
		float scaleY;
		float rotation;
		float normalBlend;
		float glossBlend;
		gfx::GfxColor albedoTint;
		gfx::GfxImage* detailMap;
		float detailHeight;
		vec2_t detailScale;
		std::byte pad_44[4];
	};
#pragma pack(pop)

	struct CamoBaseMaterial
	{
		db::xasset::MaterialDeferredHandle material;
		gfx::GfxImage* mask;
	};

#pragma pack(push, 1)
	struct CamoMaterial
	{
		std::uint16_t numBaseMaterials;
		std::byte pad_02[6];
		CamoBaseMaterial* baseMaterials;
		std::uint16_t activeChannels;
		std::byte pad_12[6];
		CamoMaterialChannel camoMaterialChannels[4];
	};

	struct WeaponCamoMaterialSet
	{
		std::uint32_t numMaterials;
		std::byte pad_04[4];
		CamoMaterial* materials;
	};

	struct WeaponCamo
	{
		const char* name;
		WeaponCamoMaterialSet* camoMaterials;
		std::uint32_t numCamoMaterials;
		std::byte pad_14[4];
	};
#pragma pack(pop)

	using weapType_t = std::int32_t;
	using weapClass_t = std::int32_t;
	using PenetrateType = std::int32_t;
	using ImpactType = std::int32_t;
	using weapInventoryType_t = std::int32_t;
	using weapFireType_t = std::int32_t;
	using weapClipType_t = std::int32_t;
	using barrelType_t = std::int32_t;
	using OffhandClass = std::int32_t;
	using OffhandSlot = std::int32_t;
	using weapStance_t = std::int32_t;
	using activeReticleType_t = std::int32_t;
	using weaponIconRatioType_t = std::int32_t;
	using ammoCounterClipType_t = std::int32_t;
	using weapProjExposion_t = std::int32_t;
	using WeapStickinessType = std::int32_t;
	using WeapStickOrientation = std::int32_t;
	using WeapRotateType = std::int32_t;
	using guidedMissileType_t = std::int32_t;
	using weapLockType_t = std::int32_t;
	using weapOverlayReticle_t = std::int32_t;
	using WeapOverlayInteface_t = std::int32_t;
	using DamageType = std::int32_t;

#pragma pack(push, 1)
	struct gadgetProperties_t
	{
		std::byte data[0x20C];
	};
#pragma pack(pop)

	struct ExplosionDamage
	{
		std::int32_t radius;
		std::int32_t height;
		std::int32_t damage;
	};

#include "boiii_weapondef.inc"

#pragma pack(push, 1)
	struct WeaponVariantDef
	{
		const char* szInternalName;
		const char* szModeIndependentName;
		std::int32_t sessionMode;
		std::int32_t iVariantCount;
		WeaponDef* weapDef;
		const char* szDisplayName;
		const char* szAltWeaponName;
		const char* szAttachmentUnique;
		WeaponAttachmentPtr* attachments;
		WeaponAttachmentUniquePtr* attachmentUniques;
		const char** szXAnims;
		scr::ScrString_t* hideTags;
		game::db::xasset::XModelPtr* attachViewModel;
		game::db::xasset::XModelPtr* attachWorldModel;
		const char** attachViewModelTag;
		const char** attachWorldModelTag;
		float attachViewModelOffsets[15];
		float attachWorldModelOffsets[15];
		float attachViewModelRotations[15];
		float attachWorldModelRotations[15];
		vec3_t stowedModelOffsets;
		vec3_t stowedModelRotations;
		std::uint32_t altWeaponIndex;
		std::byte pad_184[4];
		AttachmentMask iAttachments;
		bool bIgnoreAttachments;
		std::byte pad_191[3];
		std::int32_t iClipSize;
		std::int32_t iReloadTime;
		std::int32_t iReloadEmptyTime;
		std::int32_t iReloadQuickTime;
		std::int32_t iReloadQuickEmptyTime;
		std::int32_t iReloadSpecialComboTime;
		std::int32_t iReloadSpecialComboEmptyTime;
		std::int32_t iReloadSpecialComboQuickTime;
		std::int32_t iReloadSpecialComboQuickEmptyTime;
		std::int32_t iAdsTransInTime;
		std::int32_t iAdsTransOutTime;
		std::int32_t iAltRaiseTime;
		std::int32_t iAdsAltRaiseTime;
		std::int32_t meleeAssassinationStateTimeTransInTime;
		std::int32_t meleeAssassinationStateTimeTransOutTime;
		const char* szAmmoDisplayName;
		const char* szAmmoName;
		std::int32_t iAmmoIndex;
		std::byte pad_1E4[4];
		const char* szClipName;
		std::int32_t iClipIndex;
		float fAimAssistRangeAds;
		float fAdsSwayHorizScale;
		float fAdsSwayVertScale;
		float fkickAlignedInputScalar;
		float fkickOpposedInputScalar;
		float fAdsViewKickCenterSpeed;
		float fHipViewKickCenterSpeed;
		float fAdsFlinchScalar;
		float fAdsFiringFlinchScalar;
		float fAdsTurnRateScalar;
		float fAdsWallRunBobScalar;
		float fAdsAdditiveFallScalar;
		float fAdsAdditiveJumpScalar;
		float fAdsAdditiveJumpLandScalar;
		float fAdsZoom1_focalLength;
		float fAdsZoom1_fStop;
		float fAdsZoom2_focalLength;
		float fAdsZoom2_fStop;
		float fAdsZoom3_focalLength;
		float fAdsZoom3_fStop;
		float fAdsZoomFov1;
		float fAdsZoomFov2;
		float fAdsZoomFov3;
		float fAdsZoomInFrac;
		float fAdsZoomOutFrac;
		float fOverlayAlphaScale;
		float fOOPosAnimLength[4];
		bool bSilenced;
		bool bDualMag;
		bool bInfraRed;
		bool bTVGuided;
		std::uint32_t perks[4];
		bool bAntiQuickScope;
		std::byte pad_281[7];
		gfx::GfxImageHandle overlayMaterial;
		gfx::GfxImageHandle overlayMaterialLowRes;
		gfx::GfxImageHandle dpadIcon;
		weaponIconRatioType_t dpadIconRatio;
		bool noAmmoOnDpadIcon;
		std::byte pad_2A5[3];
		vec3_t ikLeftHandIdlePos;
		vec3_t ikLeftHandOffset;
		vec3_t ikLeftHandRotation;
		bool bUsingLeftHandProneIK;
		std::byte pad_2CD[3];
		vec3_t ikLeftHandProneOffset;
		vec3_t ikLeftHandProneRotation;
		vec3_t ikLeftHandUiViewerOffset;
		vec3_t ikLeftHandUiViewerRotation;
	};
#pragma pack(pop)

	static_assert(sizeof(gadgetProperties_t) == 0x20C);
	static_assert(sizeof(ExplosionDamage) == 0xC);
	static_assert(sizeof(CamoMaterialChannel) == 0x48);
	static_assert(sizeof(CamoBaseMaterial) == 0x10);
	static_assert(sizeof(CamoMaterial) == 0x138);
	static_assert(sizeof(WeaponCamoMaterialSet) == 0x10);
	static_assert(sizeof(WeaponCamo) == 0x18);
	static_assert(sizeof(WeaponDef) == 0x1900);
	static_assert(sizeof(WeaponVariantDef) == 0x300);
	static_assert(offsetof(WeaponVariantDef, weapDef) == 0x18);
	static_assert(offsetof(WeaponVariantDef, szDisplayName) == 0x20);
	static_assert(offsetof(WeaponVariantDef, szXAnims) == 0x48);
	static_assert(offsetof(WeaponVariantDef, iClipSize) == 0x194);
	static_assert(offsetof(WeaponVariantDef, iReloadTime) == 0x198);
	static_assert(offsetof(WeaponVariantDef, iAdsTransInTime) == 0x1B8);
	static_assert(offsetof(WeaponVariantDef, iAdsTransOutTime) == 0x1BC);
	static_assert(offsetof(WeaponDef, weapType) == 0x1DC);
	static_assert(offsetof(WeaponDef, iClipSize) == 0xBB4);
	static_assert(offsetof(WeaponDef, iReloadTime) == 0xBB8);
	static_assert(offsetof(WeaponDef, iProjectileSpeed) == 0x1228);
	static_assert(offsetof(WeaponDef, projectileModel) == 0x1250);
	static_assert(offsetof(WeaponDef, projExplosionEffect) == 0x1268);
	static_assert(offsetof(WeaponDef, projTrailEffect) == 0x13C8);
}
