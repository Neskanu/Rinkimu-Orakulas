from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase, relationship
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from datetime import datetime
from typing import Optional, List
import enum

class Base(DeclarativeBase):
    pass

# 1. Rinkimų tipų apibrėžimas
class ElectionType(str, enum.Enum):
    PARLIAMENT_MULTI = "seimas_daugiamandate"
    PARLIAMENT_SINGLE = "seimas_vienmandate"
    PRESIDENTIAL = "prezidentas"
    MUNICIPAL = "savivaldybes"
    EUROPEAN_PARLIAMENT = "ep"

# 2. Pagrindinė rinkimų lentelė
class Election(Base):
    __tablename__ = 'elections'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    year: Mapped[int] = mapped_column(Integer, index=True)
    type: Mapped[ElectionType] = mapped_column(String)
    name: Mapped[str] = mapped_column(String) # Pvz., "2024 m. Seimo rinkimai"
    
    districts: Mapped[List["District"]] = relationship(back_populates="election")
    participants: Mapped[List["Participant"]] = relationship(back_populates="election")

# 3. Apygardos (Districts)
class District(Base):
    __tablename__ = 'districts'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    election_id: Mapped[int] = mapped_column(ForeignKey("elections.id"))
    number: Mapped[Optional[str]] = mapped_column(String)
    name: Mapped[str] = mapped_column(String, index=True)
    
    election: Mapped["Election"] = relationship(back_populates="districts")
    precincts: Mapped[List["Precinct"]] = relationship(back_populates="district")

# 4. Apylinkės ir jų statistika (Precincts)
class Precinct(Base):
    __tablename__ = 'precincts'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    district_id: Mapped[int] = mapped_column(ForeignKey("districts.id"))
    number: Mapped[Optional[str]] = mapped_column(String)
    name: Mapped[str] = mapped_column(String, index=True)
    
    # Rinkėjų aktyvumo duomenys iškelti čia, nes jie priklauso apylinkei, o ne kandidatui
    registered_voters: Mapped[Optional[int]] = mapped_column(Integer) # RINKEJU_SKAICIUS
    total_participated: Mapped[Optional[int]] = mapped_column(Integer) # VISO_DALYVAVO
    valid_ballots: Mapped[Optional[int]] = mapped_column(Integer)
    invalid_ballots: Mapped[Optional[int]] = mapped_column(Integer)
    
    district: Mapped["District"] = relationship(back_populates="precincts")
    results: Mapped[List["VoteResult"]] = relationship(back_populates="precinct")

# 5. Dalyviai (Partijos sąrašai arba individualūs kandidatai)
class Participant(Base):
    __tablename__ = 'participants'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    election_id: Mapped[int] = mapped_column(ForeignKey("elections.id"))
    list_number: Mapped[Optional[str]] = mapped_column(String)
    name: Mapped[str] = mapped_column(String) # Partijos pavadinimas ARBA kandidato vardas/pavardė
    is_individual: Mapped[bool] = mapped_column(default=False) # True vienmandatėms/prezidento
    
    election: Mapped["Election"] = relationship(back_populates="participants")
    results: Mapped[List["VoteResult"]] = relationship(back_populates="participant")

# 6. Konkretūs balsavimo rezultatai
class VoteResult(Base):
    __tablename__ = 'vote_results'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    precinct_id: Mapped[int] = mapped_column(ForeignKey("precincts.id"))
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id"))
    
    votes_box: Mapped[Optional[int]] = mapped_column(Integer)  # BALSU_BALSADEZEJE
    votes_mail: Mapped[Optional[int]] = mapped_column(Integer) # BALSU_PASTU
    votes_total: Mapped[Optional[int]] = mapped_column(Integer, index=True) # BALSU_VISO
    
    precinct: Mapped["Precinct"] = relationship(back_populates="results")
    participant: Mapped["Participant"] = relationship(back_populates="results")

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
    __tablename__ = 'ensemble_configs'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String, default="Default Ensemble")
    
    # Esami svoriai
    rf_weight = Column(Float, default=0.2)
    nn_weight = Column(Float, default=0.2)
    catboost_weight = Column(Float, default=0.2)
    xgboost_weight = Column(Float, default=0.15)
    lgbm_weight = Column(Float, default=0.15)
    elasticnet_weight = Column(Float, default=0.1)
    
    # NAUJI SVORIAI (Užtikrinkite, kad Column ir Float yra importuoti viršuje!)
    dnn_weight = Column(Float, default=0.0)
    wnn_weight = Column(Float, default=0.0)
    svr_weight = Column(Float, default=0.0)
