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
package io.trino.operator.table.json;

import io.trino.json.ir.IrJsonPath;
import io.trino.metadata.ResolvedFunction;
import io.trino.spi.type.Type;

import static java.util.Objects.requireNonNull;

public record JsonTableQueryColumn(
        int outputIndex,
        ResolvedFunction function,
        // format is handled by output function
        IrJsonPath path,
        long wrapperBehavior,
        //long quotesBehavior, // resolved in the Planner considering default, goes to the output function
        long emptyBehavior,
        long errorBehavior) // resolved in the Planner considering the behavior of the enclosing json_table function
        implements JsonTableColumn
{
    public JsonTableQueryColumn
    {
        requireNonNull(function, "function is null");
        requireNonNull(path, "path is null");
    }
}
