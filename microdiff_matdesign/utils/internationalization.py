"""Internationalization (i18n) support for MicroDiff-MatDesign.

Supports multiple languages and regions for global deployment.
"""

import os
import json
import locale
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .logging_config import get_logger
from .error_handling import handle_errors, MicroDiffError


class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    ARABIC = "ar"


@dataclass
class LocaleConfig:
    """Configuration for locale settings."""
    language: str = "en"
    country: str = "US"
    encoding: str = "UTF-8"
    currency: str = "USD"
    number_format: str = "US"
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    timezone: str = "UTC"


class InternationalizationManager:
    """Manages internationalization and localization."""
    
    def __init__(self, default_language: str = "en"):
        """Initialize i18n manager.
        
        Args:
            default_language: Default language code
        """
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, str]] = {}
        self.locale_configs: Dict[str, LocaleConfig] = {}
        self.logger = get_logger('i18n.manager')
        
        # Initialize default locale configs
        self._initialize_default_locales()
        
        # Load translations
        self._load_translations()
    
    def _initialize_default_locales(self):
        """Initialize default locale configurations."""
        self.locale_configs = {
            "en": LocaleConfig("en", "US", "UTF-8", "USD", "US", "%Y-%m-%d", "%H:%M:%S", "UTC"),
            "es": LocaleConfig("es", "ES", "UTF-8", "EUR", "EU", "%d/%m/%Y", "%H:%M:%S", "Europe/Madrid"),
            "fr": LocaleConfig("fr", "FR", "UTF-8", "EUR", "EU", "%d/%m/%Y", "%H:%M:%S", "Europe/Paris"),
            "de": LocaleConfig("de", "DE", "UTF-8", "EUR", "EU", "%d.%m.%Y", "%H:%M:%S", "Europe/Berlin"),
            "ja": LocaleConfig("ja", "JP", "UTF-8", "JPY", "JP", "%Y年%m月%d日", "%H時%M分%S秒", "Asia/Tokyo"),
            "zh-CN": LocaleConfig("zh", "CN", "UTF-8", "CNY", "CN", "%Y年%m月%d日", "%H:%M:%S", "Asia/Shanghai"),
            "zh-TW": LocaleConfig("zh", "TW", "UTF-8", "TWD", "TW", "%Y年%m月%d日", "%H:%M:%S", "Asia/Taipei"),
            "ko": LocaleConfig("ko", "KR", "UTF-8", "KRW", "KR", "%Y년 %m월 %d일", "%H시 %M분 %S초", "Asia/Seoul"),
            "pt": LocaleConfig("pt", "BR", "UTF-8", "BRL", "BR", "%d/%m/%Y", "%H:%M:%S", "America/Sao_Paulo"),
            "it": LocaleConfig("it", "IT", "UTF-8", "EUR", "EU", "%d/%m/%Y", "%H:%M:%S", "Europe/Rome"),
            "ru": LocaleConfig("ru", "RU", "UTF-8", "RUB", "RU", "%d.%m.%Y", "%H:%M:%S", "Europe/Moscow"),
            "ar": LocaleConfig("ar", "SA", "UTF-8", "SAR", "SA", "%d/%m/%Y", "%H:%M:%S", "Asia/Riyadh")
        }
    
    def _load_translations(self):
        """Load translation files."""
        # Default English translations (built-in)
        self.translations["en"] = {
            # General interface
            "app.name": "MicroDiff Materials Design",
            "app.description": "Diffusion model for inverse materials design",
            "app.version": "1.0.0",
            
            # Actions
            "action.start": "Start",
            "action.stop": "Stop", 
            "action.pause": "Pause",
            "action.resume": "Resume",
            "action.cancel": "Cancel",
            "action.save": "Save",
            "action.load": "Load",
            "action.export": "Export",
            "action.import": "Import",
            "action.clear": "Clear",
            "action.reset": "Reset",
            "action.help": "Help",
            
            # Status messages
            "status.idle": "Idle",
            "status.running": "Running",
            "status.completed": "Completed",
            "status.failed": "Failed",
            "status.processing": "Processing",
            "status.loading": "Loading",
            "status.saving": "Saving",
            
            # Process parameters
            "param.laser_power": "Laser Power (W)",
            "param.scan_speed": "Scan Speed (mm/s)",
            "param.layer_thickness": "Layer Thickness (μm)",
            "param.hatch_spacing": "Hatch Spacing (μm)",
            "param.powder_bed_temp": "Powder Bed Temperature (°C)",
            "param.energy_density": "Energy Density (J/mm³)",
            
            # Materials
            "material.ti6al4v": "Ti-6Al-4V",
            "material.steel": "Steel",
            "material.aluminum": "Aluminum",
            "material.inconel": "Inconel",
            "material.titanium": "Titanium",
            
            # Error messages
            "error.invalid_input": "Invalid input provided",
            "error.file_not_found": "File not found",
            "error.permission_denied": "Permission denied",
            "error.network_error": "Network error",
            "error.timeout": "Operation timed out",
            "error.unknown": "Unknown error occurred",
            
            # Validation messages
            "validation.required": "This field is required",
            "validation.numeric": "Must be a number",
            "validation.positive": "Must be positive",
            "validation.range": "Value out of range",
            "validation.format": "Invalid format",
            
            # Units
            "unit.watts": "W",
            "unit.mm_per_second": "mm/s",
            "unit.micrometers": "μm",
            "unit.celsius": "°C",
            "unit.joules_per_mm3": "J/mm³",
            "unit.seconds": "s",
            "unit.minutes": "min",
            "unit.hours": "h",
            
            # Success messages
            "success.saved": "Successfully saved",
            "success.loaded": "Successfully loaded",
            "success.exported": "Successfully exported",
            "success.completed": "Operation completed successfully",
        }
        
        # Spanish translations
        self.translations["es"] = {
            "app.name": "Diseño de Materiales MicroDiff",
            "app.description": "Modelo de difusión para diseño inverso de materiales",
            "action.start": "Iniciar",
            "action.stop": "Detener",
            "action.pause": "Pausar", 
            "action.resume": "Reanudar",
            "action.cancel": "Cancelar",
            "action.save": "Guardar",
            "action.load": "Cargar",
            "action.export": "Exportar",
            "action.import": "Importar",
            "action.clear": "Limpiar",
            "action.reset": "Reiniciar",
            "action.help": "Ayuda",
            
            "status.idle": "Inactivo",
            "status.running": "Ejecutando",
            "status.completed": "Completado",
            "status.failed": "Fallido",
            "status.processing": "Procesando",
            "status.loading": "Cargando",
            "status.saving": "Guardando",
            
            "param.laser_power": "Potencia del Láser (W)",
            "param.scan_speed": "Velocidad de Escaneo (mm/s)",
            "param.layer_thickness": "Grosor de Capa (μm)",
            "param.hatch_spacing": "Espaciado de Rayado (μm)",
            "param.powder_bed_temp": "Temperatura del Lecho de Polvo (°C)",
            "param.energy_density": "Densidad de Energía (J/mm³)",
            
            "error.invalid_input": "Entrada inválida proporcionada",
            "error.file_not_found": "Archivo no encontrado",
            "error.permission_denied": "Permiso denegado",
            "error.network_error": "Error de red",
            "error.timeout": "Operación agotó el tiempo",
            "error.unknown": "Error desconocido ocurrido",
            
            "validation.required": "Este campo es requerido",
            "validation.numeric": "Debe ser un número",
            "validation.positive": "Debe ser positivo",
            "validation.range": "Valor fuera de rango",
            "validation.format": "Formato inválido",
            
            "success.saved": "Guardado exitosamente",
            "success.loaded": "Cargado exitosamente",
            "success.exported": "Exportado exitosamente",
            "success.completed": "Operación completada exitosamente",
        }
        
        # French translations
        self.translations["fr"] = {
            "app.name": "Conception de Matériaux MicroDiff",
            "app.description": "Modèle de diffusion pour la conception inverse de matériaux",
            "action.start": "Démarrer",
            "action.stop": "Arrêter",
            "action.pause": "Pause",
            "action.resume": "Reprendre",
            "action.cancel": "Annuler",
            "action.save": "Sauvegarder",
            "action.load": "Charger",
            "action.export": "Exporter",
            "action.import": "Importer",
            "action.clear": "Effacer",
            "action.reset": "Réinitialiser",
            "action.help": "Aide",
            
            "status.idle": "Inactif",
            "status.running": "En cours",
            "status.completed": "Terminé",
            "status.failed": "Échoué",
            "status.processing": "Traitement",
            "status.loading": "Chargement",
            "status.saving": "Sauvegarde",
            
            "param.laser_power": "Puissance Laser (W)",
            "param.scan_speed": "Vitesse de Balayage (mm/s)",
            "param.layer_thickness": "Épaisseur de Couche (μm)",
            "param.hatch_spacing": "Espacement des Hachures (μm)",
            "param.powder_bed_temp": "Température du Lit de Poudre (°C)",
            "param.energy_density": "Densité d'Énergie (J/mm³)",
            
            "error.invalid_input": "Entrée invalide fournie",
            "error.file_not_found": "Fichier non trouvé",
            "error.permission_denied": "Permission refusée",
            "error.network_error": "Erreur réseau",
            "error.timeout": "Opération expirée",
            "error.unknown": "Erreur inconnue survenue",
            
            "validation.required": "Ce champ est requis",
            "validation.numeric": "Doit être un nombre",
            "validation.positive": "Doit être positif",
            "validation.range": "Valeur hors limites",
            "validation.format": "Format invalide",
            
            "success.saved": "Sauvegardé avec succès",
            "success.loaded": "Chargé avec succès",
            "success.exported": "Exporté avec succès",
            "success.completed": "Opération terminée avec succès",
        }
        
        # German translations
        self.translations["de"] = {
            "app.name": "MicroDiff Materialdesign",
            "app.description": "Diffusionsmodell für inverses Materialdesign",
            "action.start": "Starten",
            "action.stop": "Stoppen",
            "action.pause": "Pausieren",
            "action.resume": "Fortsetzen",
            "action.cancel": "Abbrechen",
            "action.save": "Speichern",
            "action.load": "Laden",
            "action.export": "Exportieren",
            "action.import": "Importieren",
            "action.clear": "Löschen",
            "action.reset": "Zurücksetzen",
            "action.help": "Hilfe",
            
            "status.idle": "Bereit",
            "status.running": "Läuft",
            "status.completed": "Abgeschlossen",
            "status.failed": "Fehlgeschlagen",
            "status.processing": "Verarbeitung",
            "status.loading": "Lädt",
            "status.saving": "Speichert",
            
            "param.laser_power": "Laserleistung (W)",
            "param.scan_speed": "Scangeschwindigkeit (mm/s)",
            "param.layer_thickness": "Schichtdicke (μm)",
            "param.hatch_spacing": "Schraffurabstand (μm)",
            "param.powder_bed_temp": "Pulverbetttemperatur (°C)",
            "param.energy_density": "Energiedichte (J/mm³)",
            
            "error.invalid_input": "Ungültige Eingabe bereitgestellt",
            "error.file_not_found": "Datei nicht gefunden",
            "error.permission_denied": "Berechtigung verweigert",
            "error.network_error": "Netzwerkfehler",
            "error.timeout": "Operation zeitüberschreitung",
            "error.unknown": "Unbekannter Fehler aufgetreten",
            
            "validation.required": "Dieses Feld ist erforderlich",
            "validation.numeric": "Muss eine Zahl sein",
            "validation.positive": "Muss positiv sein",
            "validation.range": "Wert außerhalb des Bereichs",
            "validation.format": "Ungültiges Format",
            
            "success.saved": "Erfolgreich gespeichert",
            "success.loaded": "Erfolgreich geladen",
            "success.exported": "Erfolgreich exportiert",
            "success.completed": "Operation erfolgreich abgeschlossen",
        }
        
        # Japanese translations
        self.translations["ja"] = {
            "app.name": "MicroDiff材料設計",
            "app.description": "逆材料設計のための拡散モデル",
            "action.start": "開始",
            "action.stop": "停止",
            "action.pause": "一時停止",
            "action.resume": "再開",
            "action.cancel": "キャンセル",
            "action.save": "保存",
            "action.load": "読み込み",
            "action.export": "エクスポート",
            "action.import": "インポート",
            "action.clear": "クリア",
            "action.reset": "リセット",
            "action.help": "ヘルプ",
            
            "status.idle": "待機中",
            "status.running": "実行中",
            "status.completed": "完了",
            "status.failed": "失敗",
            "status.processing": "処理中",
            "status.loading": "読み込み中",
            "status.saving": "保存中",
            
            "param.laser_power": "レーザー出力 (W)",
            "param.scan_speed": "スキャン速度 (mm/s)",
            "param.layer_thickness": "層厚 (μm)",
            "param.hatch_spacing": "ハッチ間隔 (μm)",
            "param.powder_bed_temp": "パウダーベッド温度 (°C)",
            "param.energy_density": "エネルギー密度 (J/mm³)",
            
            "error.invalid_input": "無効な入力が提供されました",
            "error.file_not_found": "ファイルが見つかりません",
            "error.permission_denied": "アクセス権限がありません",
            "error.network_error": "ネットワークエラー",
            "error.timeout": "操作がタイムアウトしました",
            "error.unknown": "不明なエラーが発生しました",
            
            "validation.required": "この項目は必須です",
            "validation.numeric": "数値である必要があります",
            "validation.positive": "正の値である必要があります",
            "validation.range": "値が範囲外です",
            "validation.format": "無効な形式です",
            
            "success.saved": "正常に保存されました",
            "success.loaded": "正常に読み込まれました",
            "success.exported": "正常にエクスポートされました",
            "success.completed": "操作が正常に完了しました",
        }
        
        # Chinese Simplified translations
        self.translations["zh-CN"] = {
            "app.name": "MicroDiff材料设计",
            "app.description": "用于逆向材料设计的扩散模型",
            "action.start": "开始",
            "action.stop": "停止",
            "action.pause": "暂停",
            "action.resume": "继续",
            "action.cancel": "取消",
            "action.save": "保存",
            "action.load": "加载",
            "action.export": "导出",
            "action.import": "导入",
            "action.clear": "清除",
            "action.reset": "重置",
            "action.help": "帮助",
            
            "status.idle": "空闲",
            "status.running": "运行中",
            "status.completed": "已完成",
            "status.failed": "失败",
            "status.processing": "处理中",
            "status.loading": "加载中",
            "status.saving": "保存中",
            
            "param.laser_power": "激光功率 (W)",
            "param.scan_speed": "扫描速度 (mm/s)",
            "param.layer_thickness": "层厚 (μm)",
            "param.hatch_spacing": "填充间距 (μm)",
            "param.powder_bed_temp": "粉床温度 (°C)",
            "param.energy_density": "能量密度 (J/mm³)",
            
            "error.invalid_input": "提供了无效输入",
            "error.file_not_found": "文件未找到",
            "error.permission_denied": "权限被拒绝",
            "error.network_error": "网络错误",
            "error.timeout": "操作超时",
            "error.unknown": "发生未知错误",
            
            "validation.required": "此字段为必填项",
            "validation.numeric": "必须是数字",
            "validation.positive": "必须是正数",
            "validation.range": "值超出范围",
            "validation.format": "格式无效",
            
            "success.saved": "保存成功",
            "success.loaded": "加载成功",
            "success.exported": "导出成功",
            "success.completed": "操作成功完成",
        }
        
        self.logger.info(f"Loaded translations for {len(self.translations)} languages")
    
    def set_language(self, language_code: str) -> bool:
        """Set the current language.
        
        Args:
            language_code: Language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            True if language was set successfully
        """
        if language_code in self.translations:
            self.current_language = language_code
            self.logger.info(f"Language set to: {language_code}")
            return True
        else:
            self.logger.warning(f"Language {language_code} not supported, using {self.default_language}")
            self.current_language = self.default_language
            return False
    
    def get_text(self, key: str, language: Optional[str] = None) -> str:
        """Get translated text for a key.
        
        Args:
            key: Translation key
            language: Language code (uses current if None)
            
        Returns:
            Translated text or key if not found
        """
        lang = language or self.current_language
        
        # Try current language
        if lang in self.translations and key in self.translations[lang]:
            return self.translations[lang][key]
        
        # Fallback to default language
        if self.default_language in self.translations and key in self.translations[self.default_language]:
            return self.translations[self.default_language][key]
        
        # Return key if no translation found
        self.logger.warning(f"Translation not found for key: {key} in language: {lang}")
        return key
    
    def get_available_languages(self) -> Dict[str, str]:
        """Get available languages with native names.
        
        Returns:
            Dictionary of language codes to native names
        """
        return {
            "en": "English",
            "es": "Español",
            "fr": "Français", 
            "de": "Deutsch",
            "ja": "日本語",
            "zh-CN": "简体中文",
            "zh-TW": "繁體中文",
            "ko": "한국어",
            "pt": "Português",
            "it": "Italiano",
            "ru": "Русский",
            "ar": "العربية"
        }
    
    def get_locale_config(self, language: Optional[str] = None) -> LocaleConfig:
        """Get locale configuration for language.
        
        Args:
            language: Language code
            
        Returns:
            Locale configuration
        """
        lang = language or self.current_language
        return self.locale_configs.get(lang, self.locale_configs[self.default_language])
    
    def format_number(self, number: float, decimal_places: int = 2, 
                     language: Optional[str] = None) -> str:
        """Format number according to locale.
        
        Args:
            number: Number to format
            decimal_places: Number of decimal places
            language: Language code
            
        Returns:
            Formatted number string
        """
        lang = language or self.current_language
        locale_config = self.get_locale_config(lang)
        
        # Simple formatting based on locale
        if locale_config.number_format == "EU":
            # European format: 1.234,56
            formatted = f"{number:,.{decimal_places}f}"
            formatted = formatted.replace(",", "TEMP").replace(".", ",").replace("TEMP", ".")
        else:
            # US format: 1,234.56
            formatted = f"{number:,.{decimal_places}f}"
        
        return formatted
    
    def format_currency(self, amount: float, language: Optional[str] = None) -> str:
        """Format currency according to locale.
        
        Args:
            amount: Amount to format
            language: Language code
            
        Returns:
            Formatted currency string
        """
        lang = language or self.current_language
        locale_config = self.get_locale_config(lang)
        
        formatted_number = self.format_number(amount, 2, lang)
        
        # Add currency symbol
        currency_symbols = {
            "USD": "$", "EUR": "€", "JPY": "¥", "GBP": "£",
            "CNY": "¥", "KRW": "₩", "BRL": "R$", "RUB": "₽",
            "SAR": "ر.س", "TWD": "NT$"
        }
        
        symbol = currency_symbols.get(locale_config.currency, locale_config.currency)
        
        if locale_config.currency in ["USD", "GBP", "CNY", "TWD"]:
            return f"{symbol}{formatted_number}"
        else:
            return f"{formatted_number} {symbol}"
    
    def export_translations(self, file_path: str) -> bool:
        """Export translations to JSON file.
        
        Args:
            file_path: Output file path
            
        Returns:
            True if export successful
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.translations, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Translations exported to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export translations: {e}")
            return False
    
    def import_translations(self, file_path: str) -> bool:
        """Import translations from JSON file.
        
        Args:
            file_path: Input file path
            
        Returns:
            True if import successful
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                imported_translations = json.load(f)
            
            # Merge with existing translations
            for lang, translations in imported_translations.items():
                if lang not in self.translations:
                    self.translations[lang] = {}
                self.translations[lang].update(translations)
            
            self.logger.info(f"Translations imported from: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import translations: {e}")
            return False


# Global i18n manager instance
i18n_manager = InternationalizationManager()

# Convenience functions
def _(key: str, language: Optional[str] = None) -> str:
    """Get translated text (convenience function).
    
    Args:
        key: Translation key
        language: Language code
        
    Returns:
        Translated text
    """
    return i18n_manager.get_text(key, language)

def set_language(language_code: str) -> bool:
    """Set current language (convenience function).
    
    Args:
        language_code: Language code
        
    Returns:
        True if successful
    """
    return i18n_manager.set_language(language_code)

def get_available_languages() -> Dict[str, str]:
    """Get available languages (convenience function).
    
    Returns:
        Dictionary of language codes to native names
    """
    return i18n_manager.get_available_languages()

def format_number(number: float, decimal_places: int = 2, 
                 language: Optional[str] = None) -> str:
    """Format number according to locale (convenience function).
    
    Args:
        number: Number to format
        decimal_places: Number of decimal places
        language: Language code
        
    Returns:
        Formatted number string
    """
    return i18n_manager.format_number(number, decimal_places, language)

def format_currency(amount: float, language: Optional[str] = None) -> str:
    """Format currency according to locale (convenience function).
    
    Args:
        amount: Amount to format
        language: Language code
        
    Returns:
        Formatted currency string
    """
    return i18n_manager.format_currency(amount, language)


@handle_errors("detect_system_language", reraise=False)
def detect_system_language() -> str:
    """Detect system language automatically.
    
    Returns:
        Detected language code or 'en' as fallback
    """
    try:
        # Try to get system locale
        system_locale = locale.getdefaultlocale()[0]
        
        if system_locale:
            # Extract language part
            lang_code = system_locale.split('_')[0].lower()
            
            # Check if supported
            if lang_code in i18n_manager.translations:
                return lang_code
        
        # Check environment variables
        for env_var in ['LANG', 'LANGUAGE', 'LC_ALL', 'LC_MESSAGES']:
            env_value = os.environ.get(env_var)
            if env_value:
                lang_code = env_value.split('_')[0].split('.')[0].lower()
                if lang_code in i18n_manager.translations:
                    return lang_code
    
    except Exception:
        pass
    
    return "en"  # Fallback to English


# Initialize with system language
detected_lang = detect_system_language()
if detected_lang != "en":
    set_language(detected_lang)