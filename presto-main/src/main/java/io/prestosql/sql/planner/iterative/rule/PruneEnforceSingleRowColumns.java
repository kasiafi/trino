/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.prestosql.sql.planner.iterative.rule;

import io.prestosql.sql.planner.Symbol;
import io.prestosql.sql.planner.plan.EnforceSingleRowNode;
import io.prestosql.sql.planner.plan.PlanNode;

import java.util.Optional;
import java.util.Set;

import static io.prestosql.sql.planner.iterative.rule.Util.restrictChildOutputs;
import static io.prestosql.sql.planner.plan.Patterns.enforceSingleRow;

public class PruneEnforceSingleRowColumns
        extends ProjectOffPushDownRule<EnforceSingleRowNode>
{
    public PruneEnforceSingleRowColumns()
    {
        super(enforceSingleRow());
    }

    @Override
    protected Optional<PlanNode> pushDownProjectOff(Context context, EnforceSingleRowNode enforceSingleRowNode, Set<Symbol> referencedOutputs)
    {
        return restrictChildOutputs(context.getIdAllocator(), enforceSingleRowNode, referencedOutputs);
    }
}
