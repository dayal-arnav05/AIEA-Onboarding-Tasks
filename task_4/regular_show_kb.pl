% Regular Show Knowledge Base
% Facts about character roles

% Park workers
park_worker(mordecai).
park_worker(rigby).
park_worker(skips).
park_worker(muscle_man).
park_worker(hi_five_ghost).

% Boss
boss(benson).

% Park manager/owner
park_manager(pops).

% Character types
character_type(mordecai, blue_jay).
character_type(rigby, raccoon).
character_type(benson, gumball_machine).
character_type(skips, yeti).
character_type(pops, lollipop).
character_type(muscle_man, green_man).
character_type(hi_five_ghost, ghost).

% Friendship relations
friends(mordecai, rigby).
friends(rigby, mordecai).
friends(muscle_man, hi_five_ghost).
friends(hi_five_ghost, muscle_man).

% Boss relationships (who reports to whom)
reports_to(mordecai, benson).
reports_to(rigby, benson).
reports_to(skips, benson).
reports_to(muscle_man, benson).
reports_to(hi_five_ghost, benson).
reports_to(benson, pops).

% Rule: Someone is in charge of another if that person reports to them
in_charge_of(Boss, Worker) :- 
    reports_to(Worker, Boss).

% Rule: Check if two characters work together
work_together(X, Y) :- 
    park_worker(X), 
    park_worker(Y), 
    X \= Y.

% Rule: Check if someone has authority (is a boss or manager)
has_authority(X) :- boss(X).
has_authority(X) :- park_manager(X).

% Rule: Check if someone is subordinate (reports to someone)
is_subordinate(X) :- reports_to(X, _).

