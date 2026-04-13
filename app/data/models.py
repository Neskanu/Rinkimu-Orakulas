from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy import String, Integer, Float, DateTime
from datetime import datetime
from typing import Optional

class Base(DeclarativeBase):
    pass

class Mixin0:
    ID_ELECTION: Mapped[int] = mapped_column(Integer, primary_key=True)
    APYGARDOS_NR: Mapped[Optional[str]] = mapped_column(String)
    APYGARDOS_PAVADINIMAS: Mapped[Optional[str]] = mapped_column(String)
    APYLINKES_NR: Mapped[Optional[str]] = mapped_column(String)
    APYLINKES_PAVADINIMAS: Mapped[Optional[str]] = mapped_column(String)
    RINKEJU_SKAICIUS: Mapped[Optional[str]] = mapped_column(String)
    VISO_DALYVAVO: Mapped[Optional[str]] = mapped_column(String)
    BALSADEZEJE_GALIOJANTYS: Mapped[Optional[str]] = mapped_column(String)
    BALSADEZEJE_NEGALIOJANTYS: Mapped[Optional[str]] = mapped_column(String)
    PASTU_GALIOJANTYS: Mapped[Optional[str]] = mapped_column(String)
    PASTU_NEGALIOJANTYS: Mapped[Optional[str]] = mapped_column(String)
    SARASO_NUMERIS: Mapped[Optional[str]] = mapped_column(String)
    SARASO_PAVADINIMAS: Mapped[Optional[str]] = mapped_column(String)
    BALSU_BALSADEZEJE: Mapped[Optional[str]] = mapped_column(String)
    BALSU_PASTU: Mapped[Optional[str]] = mapped_column(String)
    BALSU_VISO: Mapped[Optional[str]] = mapped_column(String)
    PAPILDOMO_PROTOKOLO_POZYMIS: Mapped[Optional[str]] = mapped_column(String)
    RPL_UNIKALUS_NUMERIS: Mapped[Optional[str]] = mapped_column(String)
    SAV: Mapped[Optional[str]] = mapped_column(String)
    SUGENERAVIMO_DATA: Mapped[Optional[str]] = mapped_column(String)

class Daugiamandates2019Prezidentas(Base, Mixin0):
    __tablename__ = 'daugiamandates_2019_PREZIDENTAS'

class Daugiamandates2019(Base, Mixin0):
    __tablename__ = 'daugiamandates_2019'

class Daugiamandates2019Ep(Base, Mixin0):
    __tablename__ = 'daugiamandates_2019_EP'

class Daugiamandates2019Savivaldybes(Base, Mixin0):
    __tablename__ = 'daugiamandates_2019_SAVIVALDYBES'

class Daugiamandates2020(Base, Mixin0):
    __tablename__ = 'daugiamandates_2020'

class Daugiamandates2023Savivaldybes(Base, Mixin0):
    __tablename__ = 'daugiamandates_2023_SAVIVALDYBES'

class Daugiamandates2024Ep(Base, Mixin0):
    __tablename__ = 'daugiamandates_2024_EP'

class Daugiamandates2024(Base, Mixin0):
    __tablename__ = 'daugiamandates_2024'

class Mixin1:
    ID_ELECTION: Mapped[int] = mapped_column(Integer, primary_key=True)
    APYGARDOS_NR: Mapped[Optional[str]] = mapped_column(String)
    APYGARDOS_PAVADINIMAS: Mapped[Optional[str]] = mapped_column(String)
    APYLINKES_NR: Mapped[Optional[str]] = mapped_column(String)
    APYLINKES_PAVADINIMAS: Mapped[Optional[str]] = mapped_column(String)
    RINKEJU_SKAICIUS: Mapped[Optional[str]] = mapped_column(String)
    RPL_UNIKALUS_NUMERIS: Mapped[Optional[str]] = mapped_column(String)
    SAV: Mapped[Optional[str]] = mapped_column(String)
    BALSAVO_IKI_RINKIM_DIENOS: Mapped[Optional[str]] = mapped_column(String)
    _VAL: Mapped[Optional[str]] = mapped_column(String)
    DATA_PASIRINK_DAT_GAUSITE_AKTYVUMO_DUOMENIS_TAI_DIENAI_TOS_IR_ANKSTESNI_DIEN_SUMINIUS_DUOMENIS: Mapped[Optional[str]] = mapped_column(String)
    BALSAVO_SPEC_PUNKTE: Mapped[Optional[str]] = mapped_column(String)
    BALSAVO_NAMUOSE: Mapped[Optional[str]] = mapped_column(String)
    IS_VISO: Mapped[Optional[str]] = mapped_column(String)
    IS_VISO_PROC: Mapped[Optional[str]] = mapped_column(String)
    SUGENERAVIMO_DATA: Mapped[Optional[str]] = mapped_column(String)
    col_0_VAL: Mapped[Optional[str]] = mapped_column(String, name='0_VAL')
    BALSAVO_IKI_RINKIM_DIENOS_PROC: Mapped[Optional[str]] = mapped_column(String)
    _VAL_PROC: Mapped[Optional[str]] = mapped_column(String)

class Vienmandates20241(Base, Mixin1):
    __tablename__ = 'vienmandates_2024_1'

class Mixin2:
    ID_ELECTION: Mapped[int] = mapped_column(Integer, primary_key=True)
    APYGARDOS_NR: Mapped[Optional[str]] = mapped_column(String)
    APYGARDOS_PAVADINIMAS: Mapped[Optional[str]] = mapped_column(String)
    APYLINKES_NR: Mapped[Optional[str]] = mapped_column(String)
    APYLINKES_PAVADINIMAS: Mapped[Optional[str]] = mapped_column(String)
    RINKEJU_SKAICIUS: Mapped[Optional[str]] = mapped_column(String)
    VISO_DALYVAVO: Mapped[Optional[str]] = mapped_column(String)
    BALSADEZEJE_GALIOJANTYS: Mapped[Optional[str]] = mapped_column(String)
    BALSADEZEJE_NEGALIOJANTYS: Mapped[Optional[str]] = mapped_column(String)
    PASTU_GALIOJANTYS: Mapped[Optional[str]] = mapped_column(String)
    PASTU_NEGALIOJANTYS: Mapped[Optional[str]] = mapped_column(String)
    SARASO_NUMERIS: Mapped[Optional[str]] = mapped_column(String)
    SARASO_PAVADINIMAS: Mapped[Optional[str]] = mapped_column(String)
    BALSU_BALSADEZEJE: Mapped[Optional[str]] = mapped_column(String)
    BALSU_PASTU: Mapped[Optional[str]] = mapped_column(String)
    BALSU_VISO: Mapped[Optional[str]] = mapped_column(String)
    PAPILDOMO_PROTOKOLO_POZYMIS: Mapped[Optional[str]] = mapped_column(String)
    RPL_UNIKALUS_NUMERIS: Mapped[Optional[str]] = mapped_column(String)
    SAV: Mapped[Optional[str]] = mapped_column(String)

class Daugiamandates2016(Base, Mixin2):
    __tablename__ = 'daugiamandates_2016'

class Mixin3:
    ID_ELECTION: Mapped[int] = mapped_column(Integer, primary_key=True)
    APYGARDOS_NR: Mapped[Optional[str]] = mapped_column(String)
    APYGARDOS_PAVADINIMAS: Mapped[Optional[str]] = mapped_column(String)
    APYLINKES_NR: Mapped[Optional[str]] = mapped_column(String)
    APYLINKES_PAVADINIMAS: Mapped[Optional[str]] = mapped_column(String)
    RINKEJU_SKAICIUS: Mapped[Optional[str]] = mapped_column(String)
    VISO_DALYVAVO: Mapped[Optional[str]] = mapped_column(String)
    BALSADEZEJE_GALIOJANTYS: Mapped[Optional[str]] = mapped_column(String)
    BALSADEZEJE_NEGALIOJANTYS: Mapped[Optional[str]] = mapped_column(String)
    PASTU_GALIOJANTYS: Mapped[Optional[str]] = mapped_column(String)
    PASTU_NEGALIOJANTYS: Mapped[Optional[str]] = mapped_column(String)
    VARDAS: Mapped[Optional[str]] = mapped_column(String)
    PAVARDE: Mapped[Optional[str]] = mapped_column(String)
    PARTIJA: Mapped[Optional[str]] = mapped_column(String)
    BALSU_BALSADEZEJE: Mapped[Optional[str]] = mapped_column(String)
    BALSU_PASTU: Mapped[Optional[str]] = mapped_column(String)
    BALSU_VISO: Mapped[Optional[str]] = mapped_column(String)
    PAPILDOMO_PROTOKOLO_POZYMIS: Mapped[Optional[str]] = mapped_column(String)
    RKND_V_AR_ISSIKELE_PATS: Mapped[Optional[str]] = mapped_column(String)
    RPL_UNIKALUS_NUMERIS: Mapped[Optional[str]] = mapped_column(String)
    SAV: Mapped[Optional[str]] = mapped_column(String)
    SUGENERAVIMO_DATA: Mapped[Optional[str]] = mapped_column(String)

class Vienmandates2019Savivaldybes1(Base, Mixin3):
    __tablename__ = 'vienmandates_2019_SAVIVALDYBES_1'

class Vienmandates2019Savivaldybes2(Base, Mixin3):
    __tablename__ = 'vienmandates_2019_SAVIVALDYBES_2'

class Vienmandates20201(Base, Mixin3):
    __tablename__ = 'vienmandates_2020_1'

class Vienmandates2023Savivaldybes1(Base, Mixin3):
    __tablename__ = 'vienmandates_2023_SAVIVALDYBES_1'

class Vienmandates2023Savivaldybes2(Base, Mixin3):
    __tablename__ = 'vienmandates_2023_SAVIVALDYBES_2'

class Vienmandates20231(Base, Mixin3):
    __tablename__ = 'vienmandates_2023_1'

class Vienmandates20232(Base, Mixin3):
    __tablename__ = 'vienmandates_2023_2'

class Vienmandates2024Prezidentas2(Base, Mixin3):
    __tablename__ = 'vienmandates_2024_PREZIDENTAS_2'

class Vienmandates20242(Base, Mixin3):
    __tablename__ = 'vienmandates_2024_2'

class Mixin4:
    ID_ELECTION: Mapped[int] = mapped_column(Integer, primary_key=True)
    APYGARDOS_PAVADINIMAS: Mapped[Optional[str]] = mapped_column(String)
    VARDAS: Mapped[Optional[str]] = mapped_column(String)
    PAVARDE: Mapped[Optional[str]] = mapped_column(String)
    BALSU_VISO: Mapped[Optional[str]] = mapped_column(String)

class Vienmandates20121(Base, Mixin4):
    __tablename__ = 'vienmandates_2012_1'

class Vienmandates20122(Base, Mixin4):
    __tablename__ = 'vienmandates_2012_2'

class Mixin5:
    ID_ELECTION: Mapped[int] = mapped_column(Integer, primary_key=True)
    APYGARDOS_PAVADINIMAS: Mapped[Optional[str]] = mapped_column(String)
    SARASO_NUMERIS: Mapped[Optional[str]] = mapped_column(String)
    SARASO_PAVADINIMAS: Mapped[Optional[str]] = mapped_column(String)
    BALSU_VISO: Mapped[Optional[str]] = mapped_column(String)

class Daugiamandates2012(Base, Mixin5):
    __tablename__ = 'daugiamandates_2012'


class MLModelRegistry(Base):
    __tablename__ = "ml_model_registry"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_name: Mapped[str] = mapped_column(String)
    model_type: Mapped[str] = mapped_column(String)
    training_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    mae: Mapped[float] = mapped_column(Float)
    rmse: Mapped[float] = mapped_column(Float)
    file_path: Mapped[str] = mapped_column(String)
    parameters: Mapped[Optional[str]] = mapped_column(String)

class ForecastScenario(Base):
    __tablename__ = "forecast_scenarios"
    id: Mapped[int] = mapped_column(primary_key=True)
    scenario_name: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    model_id: Mapped[int] = mapped_column(Integer)
    input_data: Mapped[str] = mapped_column(String)
    predictions: Mapped[str] = mapped_column(String)

class EnsembleConfig(Base):
    __tablename__ = "ensemble_configs"
    id: Mapped[int] = mapped_column(primary_key=True)
    rf_weight: Mapped[float] = mapped_column(Float, default=0.33)
    nn_weight: Mapped[float] = mapped_column(Float, default=0.33)
    catboost_weight: Mapped[float] = mapped_column(Float, default=0.34)
    is_active: Mapped[bool] = mapped_column(default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
