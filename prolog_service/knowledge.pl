:- dynamic symptom/1.
:- dynamic child_age/1.
:- dynamic weight/1.
:- dynamic symptom_count/1.
:- dynamic severity_score/1.

symptom_catalog(low_weight).
symptom_catalog(stunted_growth).
symptom_catalog(swollen_belly).
symptom_catalog(frequent_illness).
symptom_catalog(pale_skin).
symptom_catalog(fatigue).
symptom_catalog(brittle_hair).
symptom_catalog(delayed_development).
symptom_catalog(skin_infections).
symptom_catalog(weak_immunity).
symptom_catalog(hair_discoloration).
symptom_catalog(loss_of_appetite).
symptom_catalog(irritability).
symptom_catalog(diarrhea).
symptom_catalog(slow_healing_wounds).

symptom_score(low_weight, 3).
symptom_score(stunted_growth, 4).
symptom_score(swollen_belly, 5).
symptom_score(frequent_illness, 3).
symptom_score(pale_skin, 2).
symptom_score(fatigue, 2).
symptom_score(brittle_hair, 2).
symptom_score(delayed_development, 4).
symptom_score(skin_infections, 2).
symptom_score(weak_immunity, 3).
symptom_score(hair_discoloration, 2).
symptom_score(loss_of_appetite, 1).
symptom_score(irritability, 1).
symptom_score(diarrhea, 2).
symptom_score(slow_healing_wounds, 2).

malnutrition_outcome(severe_acute_malnutrition, 90).
malnutrition_outcome(moderate_acute_malnutrition, 60).
malnutrition_outcome(chronic_malnutrition, 120).
malnutrition_outcome(mild_malnutrition, 30).

valid_symptoms([], [], []).
valid_symptoms([H|T], [H|KT], UT) :-
    symptom_catalog(H), !,
    valid_symptoms(T, KT, UT).
valid_symptoms([H|T], KT, [H|UT]) :-
    \+ symptom_catalog(H),
    valid_symptoms(T, KT, UT).

list_all_symptoms(L) :- findall(S, symptom(S), L).

count_symptoms(N) :-
    (symptom_count(N) ->
        true
    ;
        list_all_symptoms(L),
        length(L, N)
    ).

calculate_severity(TotalScore) :-
    (severity_score(TotalScore) ->
        true
    ;
        list_all_symptoms(Symptoms),
        calculate_score_list(Symptoms, 0, TotalScore)
    ).

calculate_score_list([], Acc, Acc).
calculate_score_list([H|T], Acc, Total) :-
    symptom_score(H, V),
    NewAcc is Acc + V,
    calculate_score_list(T, NewAcc, Total).

% Updated: Use aggregated features for inference (thresholds are examples; tune to data)
has_malnutrition(severe_acute_malnutrition) :-
    symptom_count(SC), severity_score(SS),
    SC >= 4, SS >= 12.

has_malnutrition(moderate_acute_malnutrition) :-
    symptom_count(SC), severity_score(SS),
    SC >= 2, SS >= 6, SS < 12,
    \+ has_malnutrition(severe_acute_malnutrition).

has_malnutrition(chronic_malnutrition) :-
    symptom_count(SC), severity_score(SS),
    SC >= 3, SS >= 8, SS < 12,
    \+ has_malnutrition(severe_acute_malnutrition),
    \+ has_malnutrition(moderate_acute_malnutrition).

has_malnutrition(mild_malnutrition) :-
    symptom_count(SC), severity_score(SS),
    SC >= 1, SS >= 1,
    \+ has_malnutrition(severe_acute_malnutrition),
    \+ has_malnutrition(moderate_acute_malnutrition),
    \+ has_malnutrition(chronic_malnutrition).

% Fallback for individual symptoms (original logic)
has_malnutrition(severe_acute_malnutrition) :-
    \+ symptom_count(_),
    symptom(swollen_belly),
    symptom(fatigue),
    symptom(low_weight),
    symptom(frequent_illness), !.

has_malnutrition(moderate_acute_malnutrition) :-
    \+ symptom_count(_),
    symptom(stunted_growth),
    symptom(pale_skin),
    symptom(low_weight), !.

has_malnutrition(chronic_malnutrition) :-
    \+ symptom_count(_),
    symptom(delayed_development),
    symptom(brittle_hair),
    symptom(skin_infections),
    symptom(weak_immunity), !.

has_malnutrition(mild_malnutrition) :-
    \+ symptom_count(_),
    ( symptom(loss_of_appetite)
    ; symptom(irritability)
    ; symptom(hair_discoloration)
    ; symptom(diarrhea)
    ; symptom(slow_healing_wounds)
    ), !.

all_malnutrition_types(List) :-
    findall(Type, has_malnutrition(Type), Raw),
    sort(Raw, List).

% Updated: Empty list if no individual symptoms
sorted_symptoms_by_severity_desc(Sorted) :-
    (list_all_symptoms(L), L \= [] ->
        findall(S-Score, (member(S, L), symptom_score(S, Score)), Pairs),
        keysort(Pairs, AscSorted),
        reverse(AscSorted, DescSorted),
        findall(S, member(S-_, DescSorted), Sorted)
    ;
        Sorted = []
    ).

age_weight_category(severely_underweight) :-
    child_age(A), weight(W),
    ( (A =< 2, W < 8);
      (A > 2, A =< 5, W < 12);
      (A > 5, A =< 10, W < 18);
      (A > 10, A =< 12, W < 24)
    ).

age_weight_category(moderately_underweight) :-
    child_age(A), weight(W),
    ( (A =< 2, W >= 8, W < 10);
      (A > 2, A =< 5, W >= 12, W < 15);
      (A > 5, A =< 10, W >= 18, W < 22);
      (A > 10, A =< 12, W >= 24, W < 30)
    ).

age_weight_category(normal) :-
    child_age(A), weight(W),
    ( (A =< 2, W >= 10);
      (A > 2, A =< 5, W >= 15);
      (A > 5, A =< 10, W >= 22);
      (A > 10, A =< 12, W >= 30)
    ).

severity_level(mild) :- calculate_severity(S), S =< 6, !.
severity_level(moderate) :- calculate_severity(S), S > 6, S =< 12, !.
severity_level(severe) :- calculate_severity(S), S > 12.

recommendations(severe_acute_malnutrition, [
    'Immediate hospital referral',
    'Therapeutic feeding (F-75, F-100)',
    'High-calorie and high-protein supplements'
]).

recommendations(moderate_acute_malnutrition, [
    'Nutrient-dense meals',
    'Vitamin and mineral supplementation',
    'Regular health checkups'
]).

recommendations(chronic_malnutrition, [
    'Balanced diet with micronutrients',
    'Long-term growth monitoring',
    'Immunization and deworming'
]).

recommendations(mild_malnutrition, [
    'Protein-rich diet',
    'Iron and vitamin supplements',
    'Increase meal frequency'
]).

diagnose_json(Result) :-
    all_malnutrition_types(Types),
    ( age_weight_category(Cat) -> Category = Cat ; Category = unknown ),
    count_symptoms(SymCount),
    sorted_symptoms_by_severity_desc(SortedSymptoms),
    calculate_severity(Score),
    ( severity_level(Level) -> SevLevel = Level ; SevLevel = unknown ),
    findall(R, (member(T, Types), recommendations(T, RList), member(R, RList)), RecList),
    sort(RecList, FlatRecs),
    findall(O, (member(T, Types), malnutrition_outcome(T, O)), Outcomes),
    (Outcomes = [] -> PredOutcome = unknown ; min_list(Outcomes, PredOutcome)),
    Result = diagnosis{
        weight_category: Category,
        malnutrition_types: Types,
        symptoms: SortedSymptoms,
        symptom_count: SymCount,
        severity_score: Score,
        severity_level: SevLevel,
        recommendations: FlatRecs,
        outcome: PredOutcome
    }.

diagnose_json(Result) :-
    \+ child_age(_),
    count_symptoms(SymCount),
    sorted_symptoms_by_severity_desc(SortedSymptoms),
    calculate_severity(Score),
    Result = diagnosis{
        weight_category: unknown,
        malnutrition_types: [],
        symptoms: SortedSymptoms,
        symptom_count: SymCount,
        severity_score: Score,
        severity_level: unknown,
        recommendations: ['Consult a doctor'],
        outcome: unknown
    }.

% Add diagnose_kv predicate that returns key-value list
diagnose_kv(KV) :-
    all_malnutrition_types(Types),
    ( age_weight_category(Cat) -> Category = Cat ; Category = unknown ),
    count_symptoms(SymCount),
    sorted_symptoms_by_severity_desc(SortedSymptoms),
    calculate_severity(Score),
    ( severity_level(Level) -> SevLevel = Level ; SevLevel = unknown ),
    findall(R, (member(T, Types), recommendations(T, RList), member(R, RList)), RecList),
    sort(RecList, FlatRecs),
    findall(O, (member(T, Types), malnutrition_outcome(T, O)), Outcomes),
    (Outcomes = [] -> PredOutcome = unknown ; min_list(Outcomes, PredOutcome)),
    KV = [
        weight_category(Category),
        malnutrition_types(Types),
        symptoms(SortedSymptoms),
        symptom_count(SymCount),
        severity_score(Score),
        severity_level(SevLevel),
        recommendations(FlatRecs),
        outcome(PredOutcome)
    ].

diagnose_kv(KV) :-
    \+ child_age(_),
    count_symptoms(SymCount),
    sorted_symptoms_by_severity_desc(SortedSymptoms),
    calculate_severity(Score),
    KV = [
        weight_category(unknown),
        malnutrition_types([]),
        symptoms(SortedSymptoms),
        symptom_count(SymCount),
        severity_score(Score),
        severity_level(unknown),
        recommendations(['Consult a doctor']),
        outcome(unknown)
    ].

% Treatment graph nodes and edges
node(malnourished_child, 0).
node(nutrition_assessment, 120).
node(micronutrient_supplementation, 90).
node(therapeutic_feeding, 60).
node(hospital_referral, 30).
node(nutritional_counseling, 70).
node(vitamin_supplementation, 80).
node(monitoring_growth, 40).
node(immunization, 50).
node(medical_treatment, 45).
node(healthy_child, 0).

and_or(malnourished_child, or, [nutrition_assessment, hospital_referral]).
and_or(nutrition_assessment, and, [micronutrient_supplementation, therapeutic_feeding, vitamin_supplementation]).
and_or(hospital_referral, and, [medical_treatment, immunization]).
and_or(micronutrient_supplementation, or, [monitoring_growth]).
and_or(therapeutic_feeding, or, [monitoring_growth]).
and_or(vitamin_supplementation, or, [monitoring_growth]).
and_or(medical_treatment, or, [healthy_child]).
and_or(immunization, or, [healthy_child]).
and_or(monitoring_growth, or, [healthy_child]).
and_or(healthy_child, or, []).
and_or(nutritional_counseling, or, [monitoring_growth]).

% AO* search algorithm
ao_star(Node, [Node], Cost) :-
    and_or(Node, _, []),
    node(Node, Cost).

ao_star(Node, [Node|Rest], TotalCost) :-
    and_or(Node, or, Children),
    Children \= [],
    findall(CostC-StratC, (member(C, Children), ao_star(C, StratC, CostC)), ChildCosts),
    findall(CostC, member(CostC-_, ChildCosts), AllCosts),
    min_list(AllCosts, MinCost),
    member(MinCost-Rest, ChildCosts),
    node(Node, NodeCost),
    TotalCost is NodeCost + MinCost.

ao_star(Node, [Node|Strategies], TotalCost) :-
    and_or(Node, and, Children),
    Children \= [],
    findall(CostC-StratC, (member(C, Children), ao_star(C, StratC, CostC)), ChildCosts),
    findall(CostC, member(CostC-_, ChildCosts), Costs),
    sum_list(Costs, SumCosts),
    findall(StratC, member(_-StratC, ChildCosts), Strategies),
    node(Node, NodeCost),
    TotalCost is NodeCost + SumCosts.

ao_star_with_cost(Node, Strategy, Cost) :-
    ao_star(Node, Strategy, Cost).

% A* search algorithm with heuristics
heuristic(malnourished_child, 150).
heuristic(nutrition_assessment, 120).
heuristic(micronutrient_supplementation, 90).
heuristic(therapeutic_feeding, 60).
heuristic(hospital_referral, 30).
heuristic(nutritional_counseling, 70).
heuristic(vitamin_supplementation, 80).
heuristic(monitoring_growth, 40).
heuristic(immunization, 50).
heuristic(medical_treatment, 45).
heuristic(healthy_child, 0).

edge(malnourished_child, nutrition_assessment, 120).
edge(malnourished_child, hospital_referral, 30).
edge(nutrition_assessment, micronutrient_supplementation, 90).
edge(nutrition_assessment, therapeutic_feeding, 60).
edge(nutrition_assessment, vitamin_supplementation, 80).
edge(hospital_referral, medical_treatment, 45).
edge(hospital_referral, immunization, 50).
edge(micronutrient_supplementation, monitoring_growth, 40).
edge(therapeutic_feeding, monitoring_growth, 40).
edge(vitamin_supplementation, monitoring_growth, 40).
edge(medical_treatment, healthy_child, 0).
edge(immunization, healthy_child, 0).
edge(monitoring_growth, healthy_child, 0).
edge(nutritional_counseling, monitoring_growth, 70).

a_star(Start, Goal, Path, Cost) :-
    heuristic(Start, HStart),
    a_star_search([node(Start, [], 0, HStart)], Goal, RevPath, Cost),
    reverse(RevPath, Path).

a_star_search([node(Current, Path, G, _)|_], Current, [Current|Path], G).

a_star_search([node(Current, Path, G, F)|Rest], Goal, FinalPath, FinalCost) :-
    findall(node(Next, [Current|Path], NewG, NewF),
            ( edge(Current, Next, EdgeCost),
              NewG is G + EdgeCost,
              heuristic(Next, H),
              NewF is NewG + H,
              \+ member(Next, [Current|Path])
            ),
            Successors),
    append(Rest, Successors, NewOpen),
    sort_nodes(NewOpen, SortedOpen),
    a_star_search(SortedOpen, Goal, FinalPath, FinalCost).

sort_nodes(Nodes, Sorted) :-
    sort(4, @=<, Nodes, Sorted).